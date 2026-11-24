
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from scipy import ndimage as ndi
from scipy.optimize import minimize
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI in Radiotherapy Workshop",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# COMPUTING DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SYNTHETIC IMAGE GENERATION
# ============================================================

def make_ct_phantom(size=128, target_scale=1.0, target_mask=None):

    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]

    image = np.zeros((size, size), dtype=np.float32)

    main = (x / 0.78) ** 2 + (y / 0.88) ** 2 <= 1
    image[main] = 0.45

    if target_mask is None:
        structure1 = (
            ((x + 0.28) / (0.22 * target_scale)) ** 2
            + ((y + 0.05) / (0.35 * target_scale)) ** 2
        ) <= 1
    else:
        structure1 = np.asarray(target_mask, dtype=bool)

    structure2 = (
        ((x - 0.28) / 0.22) ** 2
        + ((y + 0.05) / 0.35) ** 2
    ) <= 1

    structure3 = (
        (x / 0.16) ** 2
        + ((y - 0.40) / 0.12) ** 2
    ) <= 1

    image[structure1] = 0.75
    image[structure2] = 0.75
    image[structure3] = 0.95

    small1 = (
        ((x + 0.48) / 0.10) ** 2
        + ((y - 0.42) / 0.12) ** 2
    ) <= 1

    small2 = (
        ((x - 0.45) / 0.12) ** 2
        + ((y - 0.38) / 0.10) ** 2
    ) <= 1

    image[small1] = 0.25
    image[small2] = 0.30
    if target_mask is not None:
        image[structure1] = 0.75

    rng = np.random.default_rng(42)

    image += rng.normal(
        0, 0.015, image.shape
    ).astype(np.float32)

    return np.clip(image, 0, 1)



def make_segmentation_mask(size=128, organ_scale=1.0):

    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]

    mask = (
        ((x + 0.28) / (0.22 * organ_scale)) ** 2
        + ((y + 0.05) / (0.35 * organ_scale)) ** 2
    ) <= 1

    return mask.astype(np.uint8)


def make_synthetic_dose_distribution(
    beam_energy,
    beam_field_widths_px=(52, 52),
    beam_orientations=(90, 0),
    reference_point_xy=None,
    size=64,
    target_mask=None
):

    z, y, x = np.mgrid[
        -1:1:complex(size),
        -1:1:complex(size),
        -1:1:complex(size)
    ]

    if target_mask is None:
        target_mask = make_segmentation_mask(size=size)
    target_y, target_x = np.nonzero(target_mask)
    coordinates = np.linspace(-1, 1, size)
    if reference_point_xy is None:
        center_x = float(coordinates[target_x].mean())
        center_y = float(coordinates[target_y].mean())
    else:
        reference_x, reference_y = reference_point_xy
        center_x = float(coordinates[reference_x])
        center_y = float(coordinates[reference_y])

    relative_x = x - center_x
    relative_y = y - center_y
    energy_factor = beam_energy / 15

    penumbra_width = 0.02
    depth_factor = np.exp(-((z - 0.15) ** 2) / 50)
    beam_profiles = []

    for beam_orientation, beam_field_width_px in zip(
        beam_orientations,
        beam_field_widths_px
    ):
        angle_rad = np.deg2rad(beam_orientation)
        direction_x = np.cos(angle_rad)
        direction_y = np.sin(angle_rad)
        normal_x = -direction_y
        normal_y = direction_x
        across_beam = relative_x * normal_x + relative_y * normal_y
        along_beam = relative_x * direction_x + relative_y * direction_y
        field_half_width = beam_field_width_px / 128
        field_edge_distance = (
            field_half_width - np.abs(across_beam)
        ) / penumbra_width
        transverse_profile = 1.0 / (
            1.0 + np.exp(-np.clip(field_edge_distance, -60, 60))
        )
        along_profile = np.exp(
            -(along_beam ** 2) / (2 * 2.5 ** 2)
        )
        beam_profiles.append(transverse_profile * along_profile)

    beam_profile = np.mean(beam_profiles, axis=0)
    dose = depth_factor * beam_profile * energy_factor
    return np.clip(dose * 100, 0, 100)


def make_2d_beam_field(target_mask, beam_orientation, beam_field_width_px):

    target_y, target_x = np.nonzero(target_mask)
    center_x = float(target_x.mean())
    center_y = float(target_y.mean())
    angle_rad = np.deg2rad(beam_orientation)
    direction_x = np.cos(angle_rad)
    direction_y = np.sin(angle_rad)
    normal_x = -direction_y
    normal_y = direction_x

    grid_y, grid_x = np.mgrid[
        0:target_mask.shape[0],
        0:target_mask.shape[1]
    ]
    relative_x = grid_x - center_x
    relative_y = grid_y - center_y
    across_beam = relative_x * normal_x + relative_y * normal_y
    along_beam = relative_x * direction_x + relative_y * direction_y
    target_depth_end = float(
        along_beam[target_mask > 0].max()
    )

    return (
        (np.abs(across_beam) <= beam_field_width_px / 2)
        & (along_beam <= target_depth_end + 1)
    )


def make_ideal_reference_dose_distribution(size=64, target_mask=None):

    if target_mask is None:
        target_mask = make_segmentation_mask(size=size)
    target_mask = np.asarray(target_mask, dtype=bool)
    target_indices = np.flatnonzero(target_mask)
    inside_distance = ndi.distance_transform_edt(target_mask)
    ranked_target_indices = np.argsort(
        inside_distance[target_mask],
        kind="stable"
    )[::-1]
    covered_count = int(round(0.95 * target_indices.size))
    target_dose = np.full(target_indices.size, 94.0, dtype=float)
    target_dose[ranked_target_indices[:covered_count]] = 100.0

    outside_distance = ndi.distance_transform_edt(~target_mask)
    ideal_slice = np.zeros(target_mask.shape, dtype=float)
    ideal_slice[target_mask] = target_dose
    ideal_slice[~target_mask] = 90.0 * np.exp(
        -outside_distance[~target_mask] / 3.0
    )
    return np.broadcast_to(
        ideal_slice,
        (size, size, size)
    ).copy()


def make_dose_plan_training_case(
    rng,
    size=32,
    scale=None,
    rotation=None,
    shift_x=None,
    shift_y=None
):

    if scale is None:
        scale = float(rng.uniform(0.65, 1.15))
    if rotation is None:
        rotation = float(rng.uniform(0, 180))
    if shift_x is None:
        shift_x = float(rng.uniform(-0.08, 0.08))
    if shift_y is None:
        shift_y = float(rng.uniform(-0.08, 0.08))

    grid_y, grid_x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    center_x = -0.28 + shift_x
    center_y = -0.05 + shift_y
    delta_x = grid_x - center_x
    delta_y = grid_y - center_y
    rotation_rad = np.deg2rad(rotation)
    minor_axis = delta_x * np.cos(rotation_rad) + delta_y * np.sin(rotation_rad)
    major_axis = -delta_x * np.sin(rotation_rad) + delta_y * np.cos(rotation_rad)
    target_mask = (
        (minor_axis / (0.22 * scale)) ** 2
        + (major_axis / (0.35 * scale)) ** 2
    ) <= 1

    target_y, target_x = np.nonzero(target_mask)
    center_point = (
        int(round(target_x.mean())),
        int(round(target_y.mean()))
    )
    minor_radius = float(np.max(np.abs(minor_axis[target_mask])))
    major_radius = float(np.max(np.abs(major_axis[target_mask])))
    area_fraction = float(np.mean(target_mask))
    center_x_normalized = float(np.mean(grid_x[target_mask]))
    center_y_normalized = float(np.mean(grid_y[target_mask]))

    features = np.asarray([
        area_fraction,
        center_x_normalized,
        center_y_normalized,
        np.sin(rotation_rad),
        np.cos(rotation_rad),
        minor_radius,
        major_radius
    ], dtype=np.float32)

    angle_1 = (rotation + 90) % 360
    angle_2 = rotation % 360
    width_1 = int(np.clip(round(minor_radius * 128 + 4), 8, 64))
    width_2 = int(np.clip(round(major_radius * 128 + 4), 8, 64))
    reference_dose = make_ideal_reference_dose_distribution(
        size=size,
        target_mask=target_mask
    )
    target_volume = np.broadcast_to(target_mask, reference_dose.shape)
    reference_v95 = float(
        np.mean(reference_dose[target_volume] >= 95) * 100
    )

    dose = make_synthetic_dose_distribution(
        15,
        beam_field_widths_px=(width_1, width_2),
        beam_orientations=(angle_1, angle_2),
        reference_point_xy=center_point,
        size=size,
        target_mask=target_mask
    )
    target_v95 = float(np.mean(dose[target_volume] >= 95) * 100)
    while target_v95 < 95 and (width_1 < 64 or width_2 < 64):
        if width_1 < 64:
            width_1 = min(width_1 + 2, 64)
        if width_2 < 64:
            width_2 = min(width_2 + 2, 64)
        dose = make_synthetic_dose_distribution(
            15,
            beam_field_widths_px=(width_1, width_2),
            beam_orientations=(angle_1, angle_2),
            reference_point_xy=center_point,
            size=size,
            target_mask=target_mask
        )
        target_v95 = float(np.mean(dose[target_volume] >= 95) * 100)

    if target_v95 < 95:
        raise ValueError(
            "Could not generate a beam plan reaching at least 95% target V95."
        )

    labels = np.asarray([
        1.0,
        np.sin(np.deg2rad(angle_1)),
        np.cos(np.deg2rad(angle_1)),
        np.sin(np.deg2rad(angle_2)),
        np.cos(np.deg2rad(angle_2)),
        width_1 / 64,
        width_2 / 64
    ], dtype=np.float32)
    plan = (15, angle_1, angle_2, width_1, width_2, center_point[0], center_point[1])
    return features, labels, target_mask, reference_dose, dose, plan, target_v95, reference_v95


class BeamPlanRecommendationMLP(nn.Module):

    def __init__(self, hidden_units=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 7)
        )

    def forward(self, values):
        return self.network(values)



# ============================================================
# RIGID IMAGE TRANSFORMATIONS
# ============================================================

def apply_rigid_transform(image, tx=0, ty=0, angle=0):

    translated = ndi.shift(
        image,
        shift=(ty, tx),
        order=1,
        mode="constant",
        cval=0
    )

    transformed = ndi.rotate(
        translated,
        angle=angle,
        reshape=False,
        order=1,
        mode="constant",
        cval=0
    )

    return transformed


def generate_moving_image(fixed, tx=12, ty=-8, angle=10):

    rotated = ndi.rotate(
        fixed,
        angle=angle,
        reshape=False,
        order=1,
        mode="constant",
        cval=0
    )

    moving = ndi.shift(
        rotated,
        shift=(ty, tx),
        order=1,
        mode="constant",
        cval=0
    )

    return moving


# ============================================================
# DEFORMABLE IMAGE TRANSFORMATIONS
# ============================================================

def create_deformation_field(size, amplitude=9):

    y, x = np.mgrid[0:size, 0:size]

    center_x = size * 0.50
    center_y = size * 0.50

    sigma_x = size * 0.16
    sigma_y = size * 0.22

    gaussian = np.exp(
        -(
            ((x - center_x) ** 2 / (2 * sigma_x ** 2))
            + ((y - center_y) ** 2 / (2 * sigma_y ** 2))
        )
    )

    dx = amplitude * gaussian

    dy = (
        0.35
        * amplitude
        * gaussian
        * ((y - center_y) / (sigma_y + 1e-8))
    )

    return dx, dy


def apply_deformation(image, dx, dy):

    y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    deformed = ndi.map_coordinates(
        image,
        [
            (y - dy).ravel(),
            (x - dx).ravel()
        ],
        order=1,
        mode="constant",
        cval=0
    ).reshape(image.shape)

    return deformed


def correct_known_deformation(image, dx, dy):

    y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    corrected = ndi.map_coordinates(
        image,
        [
            (y + dy).ravel(),
            (x + dx).ravel()
        ],
        order=1,
        mode="constant",
        cval=0
    ).reshape(image.shape)

    return corrected


# ============================================================
# METRICS
# ============================================================

def mse(image1, image2):

    return float(np.mean((image1 - image2) ** 2))


def dice(mask1, mask2):

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()

    denominator = mask1.sum() + mask2.sum()

    if denominator == 0:
        return 1.0

    return 2.0 * intersection / denominator


def hausdorff_distance(mask1, mask2):

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    if not mask1.any() and not mask2.any():
        return 0.0
    if not mask1.any() or not mask2.any():
        return float("inf")

    boundary1 = mask1 & ~ndi.binary_erosion(mask1, border_value=0)
    boundary2 = mask2 & ~ndi.binary_erosion(mask2, border_value=0)

    distance_to_boundary2 = ndi.distance_transform_edt(~boundary2)
    distance_to_boundary1 = ndi.distance_transform_edt(~boundary1)

    directed_1_to_2 = distance_to_boundary2[boundary1].max()
    directed_2_to_1 = distance_to_boundary1[boundary2].max()

    return float(max(directed_1_to_2, directed_2_to_1))


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_image(image, title, cmap="gray"):

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_overlay(fixed, moving, title):

    overlay = np.zeros(
        (fixed.shape[0], fixed.shape[1], 3),
        dtype=np.float32
    )

    overlay[:, :, 1] = fixed
    overlay[:, :, 0] = moving
    overlay[:, :, 2] = moving

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(np.clip(overlay, 0, 1))
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_deformation_field(dx, dy):

    size = dx.shape[0]

    y, x = np.mgrid[0:size:8, 0:size:8]

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.quiver(
        x,
        y,
        dx[::8, ::8],
        dy[::8, ::8]
    )

    ax.set_title("Displacement field")
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def section_header(title, description=None):

    st.markdown(f"## {title}")

    if description:
        st.markdown(description)

    st.markdown("---")


# ============================================================
# AUTOMATIC RIGID REGISTRATION
# ============================================================

def automatic_rigid_registration(fixed, moving):

    def objective(parameters):

        tx, ty, angle = parameters

        corrected = apply_rigid_transform(
            moving,
            tx=tx,
            ty=ty,
            angle=angle
        )

        return mse(fixed, corrected)

    initial_parameters = [0, 0, 0]

    bounds = [
        (-25, 25),
        (-25, 25),
        (-20, 20)
    ]

    result = minimize(
        objective,
        initial_parameters,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": 150,
            "xtol": 0.01,
            "ftol": 1e-8
        }
    )

    estimated_tx, estimated_ty, estimated_angle = result.x

    registered = apply_rigid_transform(
        moving,
        tx=estimated_tx,
        ty=estimated_ty,
        angle=estimated_angle
    )

    return (
        registered,
        estimated_tx,
        estimated_ty,
        estimated_angle,
        result
    )


# ============================================================
# AI REGISTRATION: U-NET ARCHITECTURE
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.layers(x)


class RegistrationUNet(nn.Module):
    """
    Input:
        Fixed image and moving image (2 channels).

    Output:
        Horizontal and vertical displacement fields (2 channels).
    """

    def __init__(self, max_displacement=15.0):

        super().__init__()

        self.max_displacement = max_displacement

        self.enc1 = DoubleConv(2, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )

        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )

        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(
            32, 16, kernel_size=2, stride=2
        )

        self.dec1 = DoubleConv(32, 16)

        self.output_layer = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return (
            torch.tanh(self.output_layer(d1))
            * self.max_displacement
        )


class SegmentationUNet(nn.Module):
    """Small U-Net that predicts a binary mask from a CT image."""

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.output_layer = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output_layer(d1)


UNET_STEP_DESCRIPTIONS = (
    ("Input image", "A CT slice enters as a single-channel image."),
    ("Encoder 1", "Two convolutions learn local patterns such as edges and small structures."),
    ("Downsampling 1", "A 2 × 2 max pool halves the image width and height."),
    ("Encoder 2", "A second pair of convolutions learns richer features at a smaller scale."),
    ("Downsampling 2", "Another max pool compresses the representation and increases context."),
    ("Bottleneck", "At the bottom of the U, convolutions process the most compressed features."),
    ("Skip connection 2", "Encoder 2 features bypass the bottleneck, preserving 64 × 64 detail."),
    ("Upsample and concatenate", "The decoder doubles resolution and joins features from Encoder 2."),
    ("Decoder 2", "Convolutions combine context with the saved details to refine the features."),
    ("Skip connection 1", "Encoder 1 features carry fine, full-resolution detail across the U."),
    ("Upsample and concatenate", "The decoder returns to 128 × 128 and joins the Encoder 1 features."),
    ("Decoder 1", "The final convolution block refines features at the original image resolution."),
    ("Segmentation output", "A 1 × 1 convolution gives one score per pixel; sigmoid gives foreground probabilities."),
)
CONVOLUTION_KERNEL_SIZES = (3, 5)


def render_step_controls(state_key, key_prefix, step_count, label):

    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    step_index = max(0, min(st.session_state[state_key], step_count - 1))
    st.session_state[state_key] = step_index
    previous_column, status_column, next_column = st.columns([1, 1, 1])

    with previous_column:
        if st.button(
            "← Previous",
            key=f"{key_prefix}_previous",
            disabled=step_index == 0,
            width="stretch",
        ):
            step_index -= 1
            st.session_state[state_key] = step_index

    with next_column:
        if st.button(
            "Next →",
            key=f"{key_prefix}_next",
            disabled=step_index == step_count - 1,
            width="stretch",
        ):
            step_index += 1
            st.session_state[state_key] = step_index

    with status_column:
        st.markdown(f"**{label} {step_index + 1} of {step_count}**")

    return step_index


def make_unet_architecture_diagram(step_index=0):

    nodes = [
        {"label": "Input<br>128²×1", "x": 0.65, "y": 2.5, "color": "#dbeafe"},
        {"label": "Enc. 1<br>Conv×2<br>16 ch", "x": 1.85, "y": 2.5, "color": "#bfdbfe"},
        {"label": "Pool<br>2×2", "x": 3.05, "y": 1.65, "color": "#e2e8f0"},
        {"label": "Enc. 2<br>Conv×2<br>32 ch", "x": 4.25, "y": 1.65, "color": "#bfdbfe"},
        {"label": "Pool<br>2×2", "x": 5.45, "y": 0.8, "color": "#e2e8f0"},
        {"label": "Bottleneck<br>Conv×2<br>64 ch", "x": 6.65, "y": 0.8, "color": "#ddd6fe"},
        {"label": "Up + concat<br>64²", "x": 7.85, "y": 1.65, "color": "#bbf7d0"},
        {"label": "Dec. 2<br>Conv×2<br>32 ch", "x": 9.05, "y": 1.65, "color": "#bbf7d0"},
        {"label": "Up + concat<br>128²", "x": 10.25, "y": 2.5, "color": "#bbf7d0"},
        {"label": "Dec. 1<br>Conv×2<br>16 ch", "x": 11.45, "y": 2.5, "color": "#bbf7d0"},
        {"label": "Mask<br>1×1 + σ", "x": 12.65, "y": 2.5, "color": "#fed7aa"},
    ]

    steps = UNET_STEP_DESCRIPTIONS
    step_positions = [
        (nodes[index]["x"], nodes[index]["y"])
        for index in [0, 1, 2, 3, 4, 5]
    ]
    step_positions.extend(
        [
            (6.65, 2.55),
            (nodes[6]["x"], nodes[6]["y"]),
            (nodes[7]["x"], nodes[7]["y"]),
            (6.25, 3.45),
            (nodes[8]["x"], nodes[8]["y"]),
            (nodes[9]["x"], nodes[9]["y"]),
            (nodes[10]["x"], nodes[10]["y"]),
        ]
    )

    fig = go.Figure()

    for source, target in zip(nodes, nodes[1:]):
        fig.add_annotation(
            x=target["x"] - 0.51,
            y=target["y"],
            ax=source["x"] + 0.51,
            ay=source["y"],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#64748b",
        )

    skip_connections = [
        (nodes[3], nodes[7], 2.55, "skip 2"),
        (nodes[1], nodes[9], 3.45, "skip 1"),
    ]
    for source, target, route_y, label in skip_connections:
        for x0, y0, x1, y1 in [
            (source["x"], source["y"] + 0.39, source["x"], route_y),
            (source["x"], route_y, target["x"], route_y),
        ]:
            fig.add_shape(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line={"color": "#8b5cf6", "width": 2, "dash": "dash"},
                layer="below",
            )
        fig.add_annotation(
            x=target["x"],
            y=target["y"] + 0.39,
            ax=target["x"],
            ay=route_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#8b5cf6",
        )
        fig.add_annotation(
            x=(source["x"] + target["x"]) / 2,
            y=route_y + 0.07,
            text=label,
            showarrow=False,
            font={"size": 10, "color": "#7c3aed"},
        )

    for node in nodes:
        fig.add_shape(
            type="rect",
            x0=node["x"] - 0.51,
            x1=node["x"] + 0.51,
            y0=node["y"] - 0.39,
            y1=node["y"] + 0.39,
            line={"color": "#475569", "width": 1},
            fillcolor=node["color"],
            layer="below",
        )
        fig.add_annotation(
            x=node["x"],
            y=node["y"],
            text=node["label"],
            showarrow=False,
            align="center",
            font={"size": 8, "color": "#172554"},
        )

    total_steps = len(steps)
    step_index = max(0, min(step_index, total_steps - 1))
    title, explanation = steps[step_index]
    marker_x, marker_y = step_positions[step_index]
    fig.add_trace(
        go.Scatter(
            x=[marker_x],
            y=[marker_y],
            mode="markers",
            marker={"size": 23, "color": "#f97316", "line": {"color": "white", "width": 3}},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title={
            "text": f"Step {step_index + 1} of {total_steps}: {title}<br><sup>{explanation}</sup>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 17},
        },
        height=560,
        margin={"l": 10, "r": 10, "t": 105, "b": 35},
        xaxis={"range": [0, 13.3], "visible": False, "fixedrange": True},
        yaxis={"range": [0.15, 4.0], "visible": False, "fixedrange": True},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def make_convolution_diagram(image, window_index=0, kernel_size=3):

    image = np.asarray(image, dtype=np.float32)
    kernel = np.full(
        (kernel_size, kernel_size),
        1 / kernel_size**2,
        dtype=np.float32,
    )
    output_height = image.shape[0] - kernel.shape[0] + 1
    output_width = image.shape[1] - kernel.shape[1] + 1
    feature_map = np.zeros((output_height, output_width), dtype=np.float32)
    for kernel_row in range(kernel.shape[0]):
        for kernel_col in range(kernel.shape[1]):
            feature_map += (
                image[
                    kernel_row:kernel_row + output_height,
                    kernel_col:kernel_col + output_width,
                ]
                * kernel[kernel_row, kernel_col]
            )
    feature_limit = max(
        float(np.max(feature_map)),
        float(np.finfo(np.float32).eps),
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.38, 0.24, 0.38],
        horizontal_spacing=0.05,
        subplot_titles=(
            f"CT ({image.shape[1]} × {image.shape[0]})",
            f"Average kernel ({kernel_size} × {kernel_size})",
            f"Full output ({output_width} × {output_height})",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=image,
            x=np.arange(image.shape[1]),
            y=np.arange(image.shape[0]),
            zmin=float(np.min(image)),
            zmax=float(np.max(image)),
            colorscale=[[0, "#000000"], [1, "#ffffff"]],
            showscale=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=kernel,
            x=np.arange(kernel.shape[1]),
            y=np.arange(kernel.shape[0]),
            zmin=0,
            zmax=float(np.max(kernel)),
            colorscale="Blues",
            showscale=False,
            xgap=4,
            ygap=4,
            text=[[f"{value:.2f}" for value in line] for line in kernel],
            texttemplate="%{text}",
            textfont={"size": 17, "color": "#0f172a"},
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=feature_map,
            x=np.arange(output_width),
            y=np.arange(output_height),
            zmin=0,
            zmax=feature_limit,
            colorscale=[[0, "#000000"], [1, "#ffffff"]],
            showscale=False,
            hovertemplate="Local average: %{z:.2f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    grid_widths = [
        image.shape[1],
        kernel.shape[1],
        output_width,
    ]
    for column, grid_width in enumerate(grid_widths, start=1):
        fig.update_xaxes(
            range=[-0.5, grid_width - 0.5],
            visible=False,
            fixedrange=True,
            row=1,
            col=column,
        )
        fig.update_yaxes(
            autorange="reversed",
            visible=False,
            fixedrange=True,
            scaleanchor=f"x{column if column > 1 else ''}",
            scaleratio=1,
            row=1,
            col=column,
        )

    total_windows = output_height * output_width
    window_index = max(0, min(window_index, total_windows - 1))
    window_row, window_col = divmod(window_index, output_width)
    window_value = feature_map[window_row, window_col]
    current_position_shapes = [
        {
            "type": "rect",
            "xref": "x",
            "yref": "y",
            "x0": window_col - 0.5,
            "x1": window_col + kernel.shape[1] - 0.5,
            "y0": window_row - 0.5,
            "y1": window_row + kernel.shape[0] - 0.5,
            "line": {"color": "#f97316", "width": 3},
            "fillcolor": "rgba(249,115,22,0.12)",
            "layer": "above",
        },
        {
            "type": "rect",
            "xref": "x3",
            "yref": "y3",
            "x0": window_col - 0.5,
            "x1": window_col + 0.5,
            "y0": window_row - 0.5,
            "y1": window_row + 0.5,
            "line": {"color": "#f97316", "width": 3},
            "fillcolor": "rgba(249,115,22,0.12)",
            "layer": "above",
        },
    ]

    fig.update_layout(
        title={
            "text": (
                f"Kernel at row {window_row + 1}, column {window_col + 1}"
                f" → output {window_value:.2f}"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        shapes=current_position_shapes,
        height=540,
        margin={"l": 5, "r": 5, "t": 75, "b": 35},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ============================================================
# AI REGISTRATION: IMAGE WARPING
# ============================================================

def warp_image_torch(moving, flow):
    """
    Warp the moving image using a predicted displacement field.

    Flow is expressed in pixels:
        flow[:, 0] = horizontal displacement
        flow[:, 1] = vertical displacement
    """

    batch, channels, height, width = moving.shape

    y, x = torch.meshgrid(
        torch.linspace(
            -1, 1, height,
            device=moving.device
        ),

        torch.linspace(
            -1, 1, width,
            device=moving.device
        ),

        indexing="ij"
    )

    base_grid = torch.stack((x, y), dim=-1)

    base_grid = base_grid.unsqueeze(0).expand(
        batch, -1, -1, -1
    )

    displacement_x = (
        flow[:, 0] * (2.0 / (width - 1))
    )

    displacement_y = (
        flow[:, 1] * (2.0 / (height - 1))
    )

    displacement = torch.stack(
        (displacement_x, displacement_y),
        dim=-1
    )

    sampling_grid = base_grid + displacement

    return F.grid_sample(
        moving,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )


# ============================================================
# AI REGISTRATION: DATASET GENERATION
# ============================================================

def create_random_registration_pair(rng, size=64):

    fixed = make_ct_phantom(size=size).astype(np.float32)

    fixed = ndi.rotate(
        fixed,
        angle=float(rng.uniform(-5, 5)),
        reshape=False,
        order=1,
        mode="constant",
        cval=0
    )

    fixed = ndi.shift(
        fixed,
        shift=(
            float(rng.uniform(-2, 2)),
            float(rng.uniform(-2, 2))
        ),
        order=1,
        mode="constant",
        cval=0
    )

    y, x = np.mgrid[0:size, 0:size]

    dx = np.zeros((size, size), dtype=np.float32)
    dy = np.zeros((size, size), dtype=np.float32)

    for _ in range(int(rng.integers(1, 4))):

        cx = rng.uniform(size * 0.25, size * 0.75)
        cy = rng.uniform(size * 0.25, size * 0.75)

        sigma = rng.uniform(size * 0.10, size * 0.22)

        amplitude_x = rng.uniform(-5, 5)
        amplitude_y = rng.uniform(-5, 5)

        gaussian = np.exp(
            -(
                (x - cx) ** 2 + (y - cy) ** 2
            ) / (2 * sigma ** 2)
        )

        dx += amplitude_x * gaussian
        dy += amplitude_y * gaussian

    moving = ndi.map_coordinates(
        fixed,
        [
            (y - dy).ravel(),
            (x - dx).ravel()
        ],
        order=1,
        mode="constant",
        cval=0
    ).reshape(size, size)

    fixed += rng.normal(
        0, 0.008, fixed.shape
    ).astype(np.float32)

    moving += rng.normal(
        0, 0.008, moving.shape
    ).astype(np.float32)

    fixed = np.clip(fixed, 0, 1)
    moving = np.clip(moving, 0, 1)

    return fixed, moving, dx, dy


def generate_registration_dataset(
    n_samples,
    seed=42,
    size=64
):

    rng = np.random.default_rng(seed)

    fixed_images = []
    moving_images = []
    displacement_fields = []

    for _ in range(n_samples):

        fixed, moving, dx, dy = (
            create_random_registration_pair(
                rng,
                size=size
            )
        )

        fixed_images.append(fixed)
        moving_images.append(moving)

        displacement_fields.append(
            np.stack([dx, dy], axis=0)
        )

    return (
        np.asarray(fixed_images, dtype=np.float32),
        np.asarray(moving_images, dtype=np.float32),
        np.asarray(displacement_fields, dtype=np.float32)
    )


def create_random_segmentation_sample(rng, size=64, organ_scale=1.0):

    image = make_ct_phantom(size=size, target_scale=organ_scale)
    mask = make_segmentation_mask(size=size, organ_scale=organ_scale)

    angle = float(rng.uniform(-18, 18))
    shift = tuple(rng.uniform(-4, 4, size=2))

    image = ndi.rotate(
        image,
        angle=angle,
        reshape=False,
        order=1,
        mode="constant",
        cval=0
    )
    mask = ndi.rotate(
        mask,
        angle=angle,
        reshape=False,
        order=0,
        mode="constant",
        cval=0
    )

    image = ndi.shift(
        image,
        shift=shift,
        order=1,
        mode="constant",
        cval=0
    )
    mask = ndi.shift(
        mask,
        shift=shift,
        order=0,
        mode="constant",
        cval=0
    )

    image = image * rng.uniform(0.9, 1.1)
    image += rng.normal(0, 0.02, image.shape).astype(np.float32)

    return (
        np.clip(image, 0, 1).astype(np.float32),
        (mask > 0.5).astype(np.float32)
    )


def generate_segmentation_dataset(
    n_samples,
    seed=42,
    size=64,
    organ_scale=1.0
):

    rng = np.random.default_rng(seed)
    images = []
    masks = []

    for _ in range(n_samples):
        image, mask = create_random_segmentation_sample(
            rng,
            size=size,
            organ_scale=organ_scale
        )
        images.append(image)
        masks.append(mask)

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(masks, dtype=np.float32)
    )


# ============================================================
# AI REGISTRATION: LOSS FUNCTION
# ============================================================

def registration_loss(
    fixed,
    moving,
    predicted_flow,
    target_flow
):

    warped = warp_image_torch(
        moving,
        predicted_flow
    )

    flow_loss = F.mse_loss(
        predicted_flow,
        target_flow
    )

    image_loss = F.mse_loss(
        warped,
        fixed
    )

    dx = (
        predicted_flow[:, :, :, 1:]
        - predicted_flow[:, :, :, :-1]
    )

    dy = (
        predicted_flow[:, :, 1:, :]
        - predicted_flow[:, :, :-1, :]
    )

    smoothness_loss = (
        dx.square().mean()
        + dy.square().mean()
    )

    total_loss = (
        flow_loss
        + 0.5 * image_loss
        + 0.01 * smoothness_loss
    )

    return (
        total_loss,
        flow_loss,
        image_loss
    )


# ============================================================
# AI REGISTRATION: TRAINING
# ============================================================

def train_registration_model(
    model,
    train_loader,
    validation_loader,
    epochs,
    learning_rate,
    progress_bar,
    status_text
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "training": [],
        "validation": []
    }

    for epoch in range(epochs):

        model.train()

        training_losses = []

        for fixed, moving, target_flow in train_loader:

            fixed = fixed.to(DEVICE)
            moving = moving.to(DEVICE)
            target_flow = target_flow.to(DEVICE)

            inputs = torch.cat(
                [fixed, moving],
                dim=1
            )

            optimizer.zero_grad()

            predicted_flow = model(inputs)

            loss, _, _ = registration_loss(
                fixed,
                moving,
                predicted_flow,
                target_flow
            )

            loss.backward()

            optimizer.step()

            training_losses.append(loss.item())

        model.eval()

        validation_losses = []

        with torch.no_grad():

            for fixed, moving, target_flow in validation_loader:

                fixed = fixed.to(DEVICE)
                moving = moving.to(DEVICE)
                target_flow = target_flow.to(DEVICE)

                inputs = torch.cat(
                    [fixed, moving],
                    dim=1
                )

                predicted_flow = model(inputs)

                loss, _, _ = registration_loss(
                    fixed,
                    moving,
                    predicted_flow,
                    target_flow
                )

                validation_losses.append(loss.item())

        train_mean = float(np.mean(training_losses))
        validation_mean = float(np.mean(validation_losses))

        history["training"].append(train_mean)
        history["validation"].append(validation_mean)

        progress_bar.progress(
            (epoch + 1) / epochs
        )

        status_text.text(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Training loss: {train_mean:.5f} | "
            f"Validation loss: {validation_mean:.5f}"
        )

    return model, history


def segmentation_loss(logits, targets):

    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    spatial_dims = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=spatial_dims)
    denominator = (
        probabilities.sum(dim=spatial_dims)
        + targets.sum(dim=spatial_dims)
    )
    dice_loss = 1.0 - (
        (2.0 * intersection + 1.0)
        / (denominator + 1.0)
    )

    return bce_loss + dice_loss.mean()


def train_segmentation_model(
    model,
    train_loader,
    validation_loader,
    epochs,
    learning_rate,
    progress_bar,
    status_text
):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"training": [], "validation": []}

    for epoch in range(epochs):
        model.train()
        training_losses = []

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.item()))

        model.eval()
        validation_losses = []

        with torch.no_grad():
            for images, masks in validation_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                logits = model(images)
                loss = segmentation_loss(logits, masks)
                validation_losses.append(float(loss.item()))

        train_mean = float(np.mean(training_losses))
        validation_mean = float(np.mean(validation_losses))
        history["training"].append(train_mean)
        history["validation"].append(validation_mean)

        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Training loss: {train_mean:.4f} | "
            f"Validation loss: {validation_mean:.4f}"
        )

    return model, history


# ============================================================
# AI REGISTRATION: EVALUATION
# ============================================================

def evaluate_registration_model(
    model,
    fixed,
    moving,
    target_dx,
    target_dy
):

    model.eval()

    fixed_tensor = torch.from_numpy(
        fixed[None, None].astype(np.float32)
    ).to(DEVICE)

    moving_tensor = torch.from_numpy(
        moving[None, None].astype(np.float32)
    ).to(DEVICE)

    with torch.no_grad():

        inputs = torch.cat(
            [fixed_tensor, moving_tensor],
            dim=1
        )

        predicted_flow = model(inputs)

        registered = warp_image_torch(
            moving_tensor,
            predicted_flow
        )

    registered = (
        registered[0, 0].cpu().numpy()
    )

    predicted_flow = (
        predicted_flow[0].cpu().numpy()
    )

    endpoint_error = np.sqrt(
        (predicted_flow[0] - target_dx) ** 2
        + (predicted_flow[1] - target_dy) ** 2
    ).mean()

    return (
        registered,
        predicted_flow,
        float(endpoint_error)
    )


# ============================================================
# TOP NAVIGATION BAR
# ============================================================

st.markdown("### AI in Radiotherapy")
st.caption("Interactive workshop · Master 2 Medical Physics")

navigation_defaults = {
    "main_navigation": "Home",
    "registration_subpage": "Registration overview",
    "segmentation_subpage": "Manual segmentation",
    "dose_prediction_subpage": "3.1 · Beam planning experience",
}
for key, value in navigation_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.container(border=True):
    page = st.pills(
        "Choose a lab",
        [
            "Home",
            "Deep learning basics",
            "1 · Image registration",
            "2 · Organ segmentation",
            "3 · Dose prediction",
            "Research challenge",
            "Knowledge check",
        ],
        key="main_navigation",
        selection_mode="single",
        format_func=lambda lab: {
            "Deep learning basics": "AI basics",
            "1 · Image registration": "Registration",
            "2 · Organ segmentation": "Segmentation",
            "3 · Dose prediction": "Dose",
            "Research challenge": "Research",
            "Knowledge check": "Quiz",
        }.get(lab, lab),
        required=True,
        width="stretch",
        wrap=False,
    )

    if page == "1 · Image registration":
        st.pills(
            "Image registration labs",
            ["Registration overview", "AI-based registration"],
            key="registration_subpage",
            selection_mode="single",
            required=True,
            width="stretch",
            wrap=False,
        )
        if st.session_state["registration_subpage"] == "AI-based registration":
            page = "4 · AI-based registration"

    elif page == "2 · Organ segmentation":
        st.pills(
            "Organ segmentation labs",
            ["Manual segmentation", "AI segmentation"],
            key="segmentation_subpage",
            selection_mode="single",
            required=True,
            width="stretch",
            wrap=False,
        )
        if st.session_state["segmentation_subpage"] == "AI segmentation":
            page = "2 · AI segmentation"

    elif page == "3 · Dose prediction":
        st.pills(
            "Dose prediction labs",
            [
                "3.1 · Beam planning experience",
                "3.2 · AI-based dose prediction",
            ],
            key="dose_prediction_subpage",
            selection_mode="single",
            required=True,
            width="stretch",
            wrap=False,
        )
        if (
            st.session_state["dose_prediction_subpage"]
            == "3.2 · AI-based dose prediction"
        ):
            page = "3 · AI dose prediction"

st.info(
    "Educational demonstration using synthetic data; not intended for clinical use."
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.title("AI in radiotherapy", icon=":material/neurology:")
    st.subheader(
        "Interactive workshop for Master 2 Medical Physics students"
    )
    st.write(
        "Explore how artificial intelligence supports medical image "
        "registration, organ segmentation, and radiotherapy dose planning."
    )

    st.markdown("---")
    st.subheader("Explore the workshop")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🩻 Image registration")

        st.write(
            "Explore rigid registration, automatic registration "
            "and anatomical deformation."
        )

    with col2:

        st.markdown("### 🧠 Organ segmentation")

        st.write(
            "Understand organ delineation and segmentation evaluation."
        )

    with col3:

        st.markdown("### 🎯 Dose prediction")

        st.write(
            "Explore the principles of 3D dose distribution prediction."
        )

    st.markdown("---")

    st.markdown("## Learning objectives")

    st.markdown("""
    At the end of this workshop, students should be able to:

    - Explain the role of AI in radiotherapy.
    - Understand the principles of medical image registration.
    - Distinguish rigid and deformable registration.
    - Distinguish conventional automatic registration from AI-based methods.
    - Understand organ-at-risk segmentation.
    - Interpret segmentation evaluation metrics.
    - Understand the concept of 3D dose prediction.
    - Identify limitations of AI models in clinical practice.
    """)

    st.info(
        "Images and numerical results are synthetic teaching examples. "
        "They do not represent a clinically trained or validated AI system.",
        title="Educational use only",
    )


# ============================================================
# DEEP LEARNING BASICS
# ============================================================

elif page == "Deep learning basics":

    section_header(
        "Deep Learning in Medical Imaging",
        "Introduction to architectures commonly used in radiotherapy."
    )

    st.markdown("### What is deep learning?")

    st.write("""
    Deep learning is a subset of machine learning based on artificial
    neural networks containing multiple processing layers.

    These networks learn hierarchical representations from data
    and can be applied to image classification, segmentation,
    registration and dose prediction.
    """)

    st.markdown("### Typical AI workflow")

    st.caption(
        "From patient data to carefully supervised clinical use: select a "
        "stage below to explore the radiotherapy example."
    )

    workflow_stages = [
        {
            "name": "Data collection",
            "question": "What data do we need?",
            "definition": "Collect data that can teach an AI model to solve a specific problem.",
            "example": "Gather CT images, organ contours, and treatment dose distributions from previously treated patients for dose-prediction research.",
            "action": "Check image quality, label consistency, cohort coverage, and permissions for data use.",
            "gate": "Use appropriately consented, de-identified data and document the intended patient population.",
        },
        {
            "name": "Image preprocessing",
            "question": "How do we prepare the data?",
            "definition": "Clean, transform, and standardize data so it is suitable for the model.",
            "example": "Resize CT images to a common resolution, normalize intensity values, and align each image with its corresponding dose distribution.",
            "action": "Apply consistent spatial and intensity transformations while preserving image-label alignment.",
            "gate": "Use the same preprocessing at training and inference; transform images and labels consistently.",
        },
        {
            "name": "Dataset preparation",
            "question": "How do we organize the data?",
            "definition": "Separate processed patient data into subsets for learning, model selection, and final evaluation.",
            "example": "Split 300 patients by patient: 210 for training (70%), 45 for validation (15%), and 45 for testing (15%).",
            "action": "Assign patients to training, validation, and test subsets before fitting the model.",
            "gate": "Split by patient, not by slice, to prevent data leakage between subsets.",
        },
        {
            "name": "Model training",
            "question": "How does AI learn?",
            "definition": "The model adjusts its parameters using training examples to reduce prediction error.",
            "example": "A 3D U-Net learns relationships between CT images, target structures, organs at risk, and corresponding dose distributions.",
            "action": "Update model weights to reduce error on training patients.",
            "gate": "Only training data updates model weights; validation and test data must not be used for fitting.",
        },
        {
            "name": "Model validation",
            "question": "How do we improve the model?",
            "definition": "Evaluate candidate models during development to tune settings and select a checkpoint.",
            "example": "Evaluate predicted doses on validation patients after each epoch and select the best checkpoint.",
            "action": "Compare checkpoints and tune development choices without training on validation cases.",
            "gate": "Use validation results for selection; keep the test set untouched.",
        },
        {
            "name": "Model testing",
            "question": "How well does the final model perform on unseen data?",
            "definition": "Evaluate the final frozen model on data never used for training or model selection.",
            "example": "Predict dose distributions for test-set patients and compare predictions with their reference doses.",
            "action": "Run the final predictions once, without tuning the model against test results.",
            "gate": "Do not tune or retrain in response to test-set performance.",
        },
        {
            "name": "Performance evaluation",
            "question": "How do we measure the results?",
            "definition": "Measure model performance quantitatively and qualitatively using suitable metrics.",
            "example": "Review MAE, RMSE, dose-volume histogram differences, and gamma passing rate for dose prediction; use Dice when evaluating contours.",
            "action": "Calculate relevant metrics and inspect per-patient results and failure cases.",
            "gate": "Choose metrics for the task; inspect failure cases rather than relying on one average score.",
        },
        {
            "name": "Clinical validation",
            "question": "Can it be trusted in clinical practice?",
            "definition": "Assess whether the model is safe, reliable, and useful in the intended clinical setting.",
            "example": "Medical physicists and radiation oncologists assess target coverage, organ-at-risk constraints, and whether predictions can support planning.",
            "action": "Assess safety, generalizability, usability, and clinician oversight.",
            "gate": "Passing a held-out test is not clinical approval; independent clinical validation is required.",
        },
        {
            "name": "Clinical integration",
            "question": "How can we use it in practice?",
            "definition": "Integrate a clinically validated model into care with appropriate professional oversight.",
            "example": "Use an AI dose prediction as a starting point for plan optimization, review, and approval by a medical physicist.",
            "action": "Integrate the tool with clear responsibilities, review steps, and monitoring.",
            "gate": "The AI supports trained professionals; clinicians retain review and approval responsibility.",
        },
    ]

    st.markdown("#### Workflow at a glance")
    st.caption(
        "Each card is a stage in the journey. Data are split before model "
        "development; clinical integration follows clinical validation."
    )
    for row_start in range(0, len(workflow_stages), 3):
        summary_columns = st.columns(3)
        for column, stage_index in zip(
            summary_columns,
            range(row_start, min(row_start + 3, len(workflow_stages))),
        ):
            stage = workflow_stages[stage_index]
            with column.container(border=True):
                st.markdown(
                    f"**{stage_index + 1:02d} · {stage['name']}**"
                )
                st.caption(stage["question"])

    workflow_stage_index = render_step_controls(
        "workflow_journey_stage",
        "workflow_journey",
        len(workflow_stages),
        "Stage",
    )
    workflow_stage = workflow_stages[workflow_stage_index]

    st.subheader(workflow_stage["name"])
    with st.container(border=True):
        st.markdown(f"**{workflow_stage['question']}**")
        st.markdown(f"**Definition:** {workflow_stage['definition']}")
        st.markdown(f"**Radiotherapy example:** {workflow_stage['example']}")
        if workflow_stage_index == 2:
            split_columns = st.columns(3)
            for column, label, count, percentage in zip(
                split_columns,
                ["Training", "Validation", "Testing"],
                [210, 45, 45],
                [70, 15, 15],
            ):
                with column:
                    st.metric(label, f"{percentage}%", f"{count} patients")

        st.markdown("---")
        st.markdown("**What happens at this stage**")
        st.write(workflow_stage["action"])

    st.info(workflow_stage["gate"], title="Quality gate")

    if workflow_stage_index == 6:
        st.markdown("#### Example evaluation metrics")
        st.dataframe(
            [
                {
                    "Metric": "MAE",
                    "What it measures": "Average absolute prediction error",
                    "Typical use": "Dose prediction",
                },
                {
                    "Metric": "RMSE",
                    "What it measures": "Prediction error that penalizes larger errors more",
                    "Typical use": "Dose prediction",
                },
                {
                    "Metric": "Dice",
                    "What it measures": "Overlap between predicted and reference contours",
                    "Typical use": "Segmentation",
                },
                {
                    "Metric": "Gamma passing rate",
                    "What it measures": "Dose agreement under specified distance and dose criteria",
                    "Typical use": "Dose comparison",
                },
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Use dose-volume histogram differences alongside dose metrics "
            "when evaluating predicted dose distributions."
        )

    if workflow_stage_index == 2:
        st.markdown("#### Quick decision: which set selects the checkpoint?")
        st.write(
            "A colleague wants to choose the best model checkpoint. "
            "Which dataset should guide that choice?"
        )
        split_decision = st.radio(
            "Choose a dataset",
            ["Validation set", "Test set"],
            index=None,
            key="workflow_test_set_decision",
            horizontal=True,
            captions=[
                "For model selection and tuning",
                "Keep untouched until final testing",
            ],
            persist_state="session",
        )
        if split_decision == "Validation set":
            st.success(
                "Correct. Use the validation set to select the checkpoint; "
                "reserve the test set for one final evaluation.",
                title="Good choice",
            )
        elif split_decision == "Test set":
            st.warning(
                "That leaks test information into model selection and makes "
                "the final test score less trustworthy.",
                title="Test-set leakage",
            )


# ============================================================
# 1. IMAGE REGISTRATION
# ============================================================

    st.markdown("### Common architectures")

    architectures = {

        "CNN": """
        Convolutional Neural Networks extract local image features
        such as edges, textures and anatomical structures.
        """,

        "U-Net / 3D U-Net": """
        Encoder-decoder architectures with skip connections.
        They are widely used for medical image segmentation.
        """,

        "Transformers": """
        Transformer architectures use attention mechanisms to
        model relationships between different image regions.
        """,

        "Hybrid architectures": """
        Combine convolutional operations and attention mechanisms
        to exploit local and global image information.
        """
    }

    selected_architecture = st.selectbox(
        "Select an architecture",
        list(architectures.keys())
    )

    st.info(architectures[selected_architecture])

    if selected_architecture == "CNN":

        st.markdown("#### What does a convolution do?")
        kernel_size = st.segmented_control(
            "Kernel size",
            options=CONVOLUTION_KERNEL_SIZES,
            default=3,
            format_func=lambda size: f"{size} × {size}",
            key="cnn_convolution_kernel_size",
        )
        if (
            st.session_state.get("cnn_convolution_last_kernel_size")
            != kernel_size
        ):
            st.session_state["cnn_convolution_last_kernel_size"] = kernel_size
            st.session_state["cnn_convolution_step"] = 0

        st.markdown(
            "This fixed averaging kernel slides over the CT image. At each "
            "position, it multiplies the image patch by the kernel weights "
            "and sums the results to produce one output value. CNNs usually "
            "learn their kernel weights during training."
        )
        convolution_image = make_ct_phantom()
        convolution_output_height = (
            convolution_image.shape[0] - kernel_size + 1
        )
        convolution_output_width = (
            convolution_image.shape[1] - kernel_size + 1
        )
        convolution_step_count = (
            convolution_output_height * convolution_output_width
        )
        st.caption(
            "Use the slider to move the kernel across the CT image and inspect "
            "how each patch contributes to the output."
        )

        convolution_step = st.slider(
            "Kernel position",
            min_value=0,
            max_value=convolution_step_count - 1,
            key="cnn_convolution_step",
        )
        st.markdown(
            f"**Position {convolution_step + 1:,} "
            f"of {convolution_step_count:,}**"
        )
        st.plotly_chart(
            make_convolution_diagram(
                convolution_image,
                convolution_step,
                kernel_size,
            ),
            width="stretch",
            config={"displayModeBar": False},
            alt=f"The full CT, a {kernel_size}-by-{kernel_size} averaging kernel highlighted at its current position, and the matching output pixel",
        )

        st.caption(
            f"The orange outlines link the current {kernel_size} × "
            f"{kernel_size} CT patch to its matching output pixel. The full "
            f"output map is {convolution_output_width} × "
            f"{convolution_output_height} because no padding is added."
        )

    if selected_architecture == "U-Net / 3D U-Net":

        st.markdown("#### How a U-Net processes an image")
        st.caption(
            "Use Previous and Next to move the orange marker through the network."
        )
        unet_stage = render_step_controls(
            "unet_architecture_stage",
            "unet_architecture",
            len(UNET_STEP_DESCRIPTIONS),
            "Stage",
        )
        st.plotly_chart(
            make_unet_architecture_diagram(unet_stage),
            width="stretch",
            config={"displayModeBar": False},
            alt="U-Net diagram highlighting the selected encoding, skip-connection, decoding, or output stage",
        )
        st.caption(
            "This simplified 2D diagram follows the segmentation U-Net used "
            "in this workshop. A 3D U-Net applies the same ideas across "
            "volumetric image slices."
        )

elif page == "1 · Image registration":

    section_header(
        "Laboratory 1.1: Medical Image Registration",
        """
        Explore rigid registration, automatic registration and
        the limitations of rigid transformations when anatomy deforms.
        """
    )

    st.markdown("### Learning objectives")

    st.markdown("""
    - Understand fixed and moved images.
    - Explore rigid transformations.
    - Compare manual and automatic rigid registration.
    - Understand why rigid registration cannot correct local anatomical deformation.
    - Explore the principle of deformable registration.
    """)

    fixed_image = make_ct_phantom()

    moving_image = generate_moving_image(
        fixed_image,
        tx=12,
        ty=-8,
        angle=10
    )

    initial_error = mse(
        fixed_image,
        moving_image
    )

    st.markdown("---")
    st.markdown("## Initial images")

    st.write("""
    The fixed image is the reference image. The moved image
    has been displaced and rotated to simulate a change in
    patient position between two imaging acquisitions.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        plot_image(fixed_image, "Fixed image")

    with col2:
        plot_image(moving_image, "Moved image")

    with col3:
        plot_overlay(
            fixed_image,
            moving_image,
            "Overlayed images before registration"
        )

    st.metric("Initial MSE", f"{initial_error:.5f}")

    # --------------------------------------------------------
    # METHOD 1: MANUAL RIGID REGISTRATION
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## Method 1: Manual rigid registration")

    st.write("""
    Students manually adjust the translation and rotation
    parameters to align the moved image with the fixed image.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:

        manual_tx = st.slider(
            "Translation X (pixels)",
            -30, 30, 0,
            key="manual_tx"
        )

    with col2:

        manual_ty = st.slider(
            "Translation Y (pixels)",
            -30, 30, 0,
            key="manual_ty"
        )

    with col3:

        manual_angle = st.slider(
            "Rotation angle (degrees)",
            -15, 15, 0,
            key="manual_angle"
        )

    manual_registered = apply_rigid_transform(
        moving_image,
        tx=manual_tx,
        ty=manual_ty,
        angle=manual_angle
    )

    manual_error = mse(
        fixed_image,
        manual_registered
    )

    st.markdown("### Manual registration results")

    col1, col2, col3 = st.columns(3)

    with col1:
        plot_image(fixed_image, "Fixed image")

    with col2:
        plot_image(
            manual_registered,
            "Moved image after manual registration"
        )

    with col3:
        plot_overlay(
            fixed_image,
            manual_registered,
            "Overlayed images after manual registration"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Initial MSE", f"{initial_error:.5f}")

    with col2:
        st.metric(
            "Manual registration MSE",
            f"{manual_error:.5f}",
            delta=f"{manual_error - initial_error:.5f}",
            delta_color="inverse"
        )

    if manual_error < initial_error:

        st.success(
            "Manual registration improved image alignment."
        )

    else:

        st.warning(
            "The current transformation did not improve alignment. "
            "Try adjusting the parameters."
        )

    st.info(
        "Student activity: adjust the sliders and observe how "
        "the overlay and MSE change."
    )

    # --------------------------------------------------------
    # METHOD 2: AUTOMATIC RIGID REGISTRATION
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## Method 2: Automatic rigid registration")

    st.write("""
    An optimization algorithm estimates the transformation
    parameters without requiring students to adjust the sliders.
    It searches for the parameters that minimize the image MSE.
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Transformation model:** Rigid")
        st.write("- Translation X")
        st.write("- Translation Y")
        st.write("- Rotation")

    with col2:

        st.write("**Optimization configuration:**")
        st.write("- Algorithm: Powell")
        st.write("- Similarity metric: MSE")
        st.write("- Maximum iterations: 150")

    if st.button(
        "Run automatic registration",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Estimating rigid transformation..."):

            (
                automatic_registered,
                automatic_tx,
                automatic_ty,
                automatic_angle,
                optimization_result
            ) = automatic_rigid_registration(
                fixed_image,
                moving_image
            )

            st.session_state["automatic_registered"] = automatic_registered
            st.session_state["automatic_tx"] = automatic_tx
            st.session_state["automatic_ty"] = automatic_ty
            st.session_state["automatic_angle"] = automatic_angle
            st.session_state["automatic_result"] = optimization_result

    if "automatic_registered" in st.session_state:

        automatic_registered = st.session_state[
            "automatic_registered"
        ]

        automatic_tx = st.session_state["automatic_tx"]
        automatic_ty = st.session_state["automatic_ty"]
        automatic_angle = st.session_state["automatic_angle"]

        optimization_result = st.session_state["automatic_result"]

        automatic_error = mse(
            fixed_image,
            automatic_registered
        )

        st.markdown("### Estimated transformation parameters")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Estimated X",
                f"{automatic_tx:.2f} pixels"
            )

        with col2:

            st.metric(
                "Estimated Y",
                f"{automatic_ty:.2f} pixels"
            )

        with col3:

            st.metric(
                "Estimated rotation",
                f"{automatic_angle:.2f}°"
            )

        st.markdown("### Automatic registration results")

        col1, col2, col3 = st.columns(3)

        with col1:
            plot_image(fixed_image, "Fixed image")

        with col2:
            plot_image(
                automatic_registered,
                "Moved image after automatic registration"
            )

        with col3:
            plot_overlay(
                fixed_image,
                automatic_registered,
                "Overlayed images after automatic registration"
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Initial MSE", f"{initial_error:.5f}")

        with col2:
            st.metric(
                "Automatic registration MSE",
                f"{automatic_error:.5f}",
                delta=f"{automatic_error - initial_error:.5f}",
                delta_color="inverse"
            )

        st.write(
            "Optimization status: "
            + (
                "Successful"
                if optimization_result.success
                else "Not successful"
            )
        )

        st.write(
            f"Number of iterations: {optimization_result.nit}"
        )

        st.markdown("### Manual versus automatic comparison")

        comparison_data = {
            "Method": [
                "Before registration",
                "Manual registration",
                "Automatic registration"
            ],
            "MSE": [
                initial_error,
                manual_error,
                automatic_error
            ]
        }

        st.dataframe(
            comparison_data,
            use_container_width=True,
            hide_index=True
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Initial", "Manual", "Automatic"],
                y=[
                    initial_error,
                    manual_error,
                    automatic_error
                ],
                text=[
                    f"{initial_error:.5f}",
                    f"{manual_error:.5f}",
                    f"{automatic_error:.5f}"
                ],
                textposition="outside"
            )
        )

        fig.update_layout(
            title="Comparison of registration errors",
            xaxis_title="Registration method",
            yaxis_title="Mean Squared Error",
            height=400,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Click 'Run automatic registration' to start "
            "the automatic estimation process."
        )

    # --------------------------------------------------------
    # METHOD 3: ANATOMICAL DEFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## Method 3: Why rigid registration is insufficient "
        "for anatomical deformation"
    )

    st.write("""
    Rigid registration assumes that the anatomy behaves as a single
    object. All image points undergo the same global transformation:
    translation and rotation.

    However, organs can change shape and position locally. Examples
    include changes in bladder or rectum filling, respiratory motion,
    and anatomical changes during a radiotherapy treatment course.

    In these situations, a single rigid transformation cannot
    correct all local anatomical differences.
    """)

    st.markdown("### Step 1: Generate a locally deformed image")

    deformation_amplitude = st.slider(
        "Local deformation amplitude (pixels)",
        min_value=2,
        max_value=15,
        value=9,
        key="deformation_amplitude"
    )

    dx, dy = create_deformation_field(
        fixed_image.shape[0],
        amplitude=deformation_amplitude
    )

    deformed_image = apply_deformation(
        fixed_image,
        dx,
        dy
    )

    deformed_error = mse(
        fixed_image,
        deformed_image
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        plot_image(fixed_image, "Fixed image")

    with col2:
        plot_image(
            deformed_image,
            "Moved image with local deformation"
        )

    with col3:
        plot_overlay(
            fixed_image,
            deformed_image,
            "Overlay before correction"
        )

    st.metric(
        "MSE before correction",
        f"{deformed_error:.5f}"
    )

    st.markdown("### Step 2: Observe the deformation field")

    st.write("""
    The arrows represent local displacement vectors. Their
    directions and magnitudes vary across the image.
    """)

    plot_deformation_field(dx, dy)

    # --------------------------------------------------------
    # APPLY RIGID REGISTRATION TO DEFORMED IMAGE
    # --------------------------------------------------------

    st.markdown("### Step 3: Try rigid registration")

    if st.button(
        "Apply rigid registration to deformed image",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Performing rigid registration..."):

            (
                rigid_deformed_registered,
                rigid_tx,
                rigid_ty,
                rigid_angle,
                rigid_result
            ) = automatic_rigid_registration(
                fixed_image,
                deformed_image
            )

            st.session_state["deformation_rigid_image"] = (
                rigid_deformed_registered
            )

            st.session_state["deformation_rigid_tx"] = rigid_tx
            st.session_state["deformation_rigid_ty"] = rigid_ty
            st.session_state["deformation_rigid_angle"] = rigid_angle
            st.session_state["deformation_rigid_amplitude"] = deformation_amplitude

    rigid_is_current = (
        "deformation_rigid_image" in st.session_state
        and st.session_state.get("deformation_rigid_amplitude")
        == deformation_amplitude
    )

    if rigid_is_current:

        rigid_deformed_registered = st.session_state[
            "deformation_rigid_image"
        ]

        rigid_deformation_error = mse(
            fixed_image,
            rigid_deformed_registered
        )

        st.markdown("### Results after rigid registration")

        col1, col2, col3 = st.columns(3)

        with col1:
            plot_image(fixed_image, "Fixed image")

        with col2:
            plot_image(
                rigid_deformed_registered,
                "Image after rigid registration"
            )

        with col3:
            plot_overlay(
                fixed_image,
                rigid_deformed_registered,
                "Overlay after rigid registration"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Estimated X",
                f"{st.session_state['deformation_rigid_tx']:.2f} px"
            )

        with col2:
            st.metric(
                "Estimated Y",
                f"{st.session_state['deformation_rigid_ty']:.2f} px"
            )

        with col3:
            st.metric(
                "Estimated rotation",
                f"{st.session_state['deformation_rigid_angle']:.2f}°"
            )

        st.metric(
            "MSE after rigid registration",
            f"{rigid_deformation_error:.5f}",
            delta=f"{rigid_deformation_error - deformed_error:.5f}",
            delta_color="inverse"
        )

        st.warning("""
        Rigid registration can correct some global displacement,
        but it cannot independently move different anatomical regions.
        """)

    else:

        st.info(
            "Click 'Apply rigid registration to deformed image' "
            "to observe the limitations of rigid registration."
        )

    # --------------------------------------------------------
    # ILLUSTRATIVE DEFORMABLE CORRECTION
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### Step 4: Illustrate deformable registration")

    st.write("""
    In this synthetic example, we use the known deformation field
    to illustrate how local correction can improve alignment.
    This is not an estimated registration field.
    """)

    if st.button(
        "Apply illustrative deformable correction",
        use_container_width=True
    ):

        corrected_deformable = correct_known_deformation(
            deformed_image,
            dx,
            dy
        )

        st.session_state["deformable_corrected"] = corrected_deformable
        st.session_state["deformable_amplitude"] = deformation_amplitude

    deformable_is_current = (
        "deformable_corrected" in st.session_state
        and st.session_state.get("deformable_amplitude")
        == deformation_amplitude
    )

    if deformable_is_current:

        corrected_deformable = st.session_state[
            "deformable_corrected"
        ]

        deformable_error = mse(
            fixed_image,
            corrected_deformable
        )

        st.markdown("### Results after deformable correction")

        col1, col2, col3 = st.columns(3)

        with col1:
            plot_image(fixed_image, "Fixed image")

        with col2:
            plot_image(
                corrected_deformable,
                "Image after local correction"
            )

        with col3:
            plot_overlay(
                fixed_image,
                corrected_deformable,
                "Overlay after local correction"
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "MSE before correction",
                f"{deformed_error:.5f}"
            )

        with col2:
            st.metric(
                "MSE after local correction",
                f"{deformable_error:.5f}",
                delta=f"{deformable_error - deformed_error:.5f}",
                delta_color="inverse"
            )

        st.success(
            "The local correction illustrates the use of "
            "spatially varying transformations."
        )

        if rigid_is_current:

            rigid_deformation_error = mse(
                fixed_image,
                st.session_state["deformation_rigid_image"]
            )

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=[
                        "Before correction",
                        "Rigid registration",
                        "Local deformable correction"
                    ],
                    y=[
                        deformed_error,
                        rigid_deformation_error,
                        deformable_error
                    ],
                    text=[
                        f"{deformed_error:.5f}",
                        f"{rigid_deformation_error:.5f}",
                        f"{deformable_error:.5f}"
                    ],
                    textposition="outside"
                )
            )

            fig.update_layout(
                title="Image similarity error comparison",
                xaxis_title="Method",
                yaxis_title="MSE",
                height=400,
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "Click 'Apply illustrative deformable correction' "
            "to see the effect of a local transformation."
        )

    # --------------------------------------------------------
    # EDUCATIONAL COMPARISON
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## Rigid versus deformable registration")

    st.markdown("""
    | Characteristic | Rigid registration | Deformable registration |
    |---|---|---|
    | Transformation | Global | Spatially varying |
    | Translation | Yes | Local displacement possible |
    | Rotation | Yes | Local changes in orientation possible |
    | Shape changes | Cannot represent them | Can represent them |
    | Local anatomical movement | Cannot model independently | Can model locally |
    """)

    st.markdown("### Clinical examples")

    st.markdown("""
    **Rigid registration:**

    - Initial patient positioning.
    - Global alignment when anatomy is approximately unchanged.

    **Deformable registration:**

    - Changes in bladder and rectum filling.
    - Respiratory-related anatomical motion.
    - Changes in organ shape between treatment fractions.
    - Longitudinal anatomical changes during radiotherapy.
    """)

    st.info(
        "The deformable correction in this laboratory uses the known synthetic "
        "displacement field. Real algorithms must estimate transformations "
        "from image data while controlling anatomical plausibility and regularity.",
        title="Scientific note",
    )

    st.markdown("### Questions for students")

    st.markdown("""
    1. Why can rigid registration not correct local deformation?
    2. What is the difference between global and local transformations?
    3. Give two examples of anatomical changes during radiotherapy.
    4. Why is deformable registration useful in adaptive radiotherapy?
    5. Can a deformable registration algorithm produce unrealistic results?
    6. Why should displacement fields be evaluated?
    """)

    st.caption(
        "Synthetic educational examples only. No clinical patient data are used."
    )


# ============================================================
# 4. AI-BASED IMAGE REGISTRATION
# ============================================================

elif page == "4 · AI-based registration":

    section_header(
        "Laboratory 1.2: AI-Based Deformable Image Registration",
        """
        Train a U-Net to learn local displacement fields from
        synthetic medical image pairs and evaluate it on an unseen example.
        """
    )

    st.markdown("### Learning objectives")

    st.markdown("""
    - Understand how a neural network learns image transformations.
    - Generate a synthetic training dataset.
    - Train a U-Net using PyTorch.
    - Visualize training and validation losses.
    - Evaluate registration on an unseen image pair.
    - Compare AI registration with the initial image alignment.
    """)

    st.info(
        "This is a supervised deep-learning experiment. "
        "The model starts with random weights and learns from "
        "synthetic training examples. It is not clinically validated."
    )

    st.markdown("---")
    st.markdown("## 1. Configure the experiment")

    col1, col2, col3 = st.columns(3)

    with col1:

        n_samples = st.selectbox(
            "Training samples",
            [40, 80, 120, 200],
            index=1
        )

    with col2:

        epochs = st.selectbox(
            "Training epochs",
            [5, 10, 20, 30],
            index=1
        )

    with col3:

        learning_rate = st.selectbox(
            "Learning rate",
            [0.001, 0.0005, 0.0001],
            index=1
        )

    batch_size = st.selectbox(
        "Batch size",
        [4, 8, 16],
        index=1
    )

    st.write(f"**Computing device:** `{DEVICE}`")

    if DEVICE.type == "cuda":
        st.success("CUDA GPU detected.")
    else:
        st.info(
            "Training will use the CPU. Training time depends "
            "on your computer and selected parameters."
        )

    # --------------------------------------------------------
    # DATA GENERATION AND TRAINING
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 2. Generate the training dataset")

    if st.button(
        "Generate dataset and train U-Net",
        type="primary",
        use_container_width=True
    ):

        torch.manual_seed(42)
        np.random.seed(42)

        with st.spinner("Generating synthetic image pairs..."):

            train_fixed, train_moving, train_fields = (
                generate_registration_dataset(
                    n_samples,
                    seed=42
                )
            )

            val_fixed, val_moving, val_fields = (
                generate_registration_dataset(
                    max(20, n_samples // 4),
                    seed=123
                )
            )

        st.success(
            f"Generated {len(train_fixed)} training pairs and "
            f"{len(val_fixed)} validation pairs."
        )

        train_fixed_t = torch.from_numpy(
            train_fixed[:, None]
        )

        train_moving_t = torch.from_numpy(
            train_moving[:, None]
        )

        train_fields_t = torch.from_numpy(
            train_fields
        )

        val_fixed_t = torch.from_numpy(
            val_fixed[:, None]
        )

        val_moving_t = torch.from_numpy(
            val_moving[:, None]
        )

        val_fields_t = torch.from_numpy(
            val_fields
        )

        train_dataset = TensorDataset(
            train_fixed_t,
            train_moving_t,
            train_fields_t
        )

        validation_dataset = TensorDataset(
            val_fixed_t,
            val_moving_t,
            val_fields_t
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # ----------------------------------------------------
        # INITIALIZE MODEL
        # ----------------------------------------------------

        model = RegistrationUNet().to(DEVICE)

        st.markdown("---")
        st.markdown("## 3. Train the neural network")

        progress_bar = st.progress(0)

        status_text = st.empty()

        with st.spinner("Training U-Net..."):

            model, history = train_registration_model(
                model,
                train_loader,
                validation_loader,
                epochs,
                learning_rate,
                progress_bar,
                status_text
            )

        st.session_state["registration_model"] = model
        st.session_state["registration_history"] = history

        st.session_state["registration_training_samples"] = n_samples
        st.session_state["registration_epochs"] = epochs

        # ----------------------------------------------------
        # TRAINING CURVES
        # ----------------------------------------------------

        st.markdown("### Training and validation curves")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=history["training"],
                mode="lines+markers",
                name="Training loss"
            )
        )

        fig.add_trace(
            go.Scatter(
                y=history["validation"],
                mode="lines+markers",
                name="Validation loss"
            )
        )

        fig.update_layout(
            title="U-Net learning curves",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # TEST ON AN UNSEEN IMAGE
        # ----------------------------------------------------

        st.markdown("---")
        st.markdown("## 4. Test the trained model")

        test_fixed, test_moving, test_fields = (
            generate_registration_dataset(
                1,
                seed=999
            )
        )

        fixed = test_fixed[0]
        moving = test_moving[0]

        target_dx = test_fields[0, 0]
        target_dy = test_fields[0, 1]

        registered, predicted_flow, endpoint_error = (
            evaluate_registration_model(
                model,
                fixed,
                moving,
                target_dx,
                target_dy
            )
        )

        st.session_state["registration_test"] = {
            "fixed": fixed,
            "moving": moving,
            "registered": registered,
            "predicted_flow": predicted_flow,
            "target_dx": target_dx,
            "target_dy": target_dy,
            "endpoint_error": endpoint_error
        }

    # --------------------------------------------------------
    # DISPLAY RESULTS AFTER TRAINING
    # --------------------------------------------------------

    if "registration_test" in st.session_state:

        results = st.session_state["registration_test"]

        fixed = results["fixed"]
        moving = results["moving"]
        registered = results["registered"]

        st.markdown("---")
        st.markdown("## 5. Evaluate the AI registration")

        col1, col2, col3 = st.columns(3)

        with col1:
            plot_image(fixed, "Fixed image")

        with col2:
            plot_image(moving, "Moving image")

        with col3:
            plot_overlay(
                fixed,
                moving,
                "Overlay before AI registration"
            )

        st.markdown("### Results after AI registration")

        col1, col2, col3 = st.columns(3)

        with col1:
            plot_image(fixed, "Fixed image")

        with col2:
            plot_image(
                registered,
                "AI-registered image"
            )

        with col3:
            plot_overlay(
                fixed,
                registered,
                "Overlay after AI registration"
            )

        initial_mse = mse(fixed, moving)
        final_mse = mse(fixed, registered)

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Initial MSE",
                f"{initial_mse:.5f}"
            )

        with col2:

            st.metric(
                "AI registration MSE",
                f"{final_mse:.5f}",
                delta=f"{final_mse - initial_mse:.5f}",
                delta_color="inverse"
            )

        with col3:

            st.metric(
                "Mean displacement endpoint error",
                f"{results['endpoint_error']:.3f} pixels"
            )

        # ----------------------------------------------------
        # DISPLACEMENT FIELD VISUALIZATION
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            "## 6. Visualize the predicted displacement field"
        )

        st.write("""
        The network predicts two displacement components:
        horizontal displacement (dx) and vertical displacement (dy).
        """)

        predicted_flow = results["predicted_flow"]

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Predicted displacement field**")

            plot_deformation_field(
                predicted_flow[0],
                predicted_flow[1]
            )

        with col2:

            st.markdown("**Reference displacement field**")

            plot_deformation_field(
                results["target_dx"],
                results["target_dy"]
            )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.markdown("---")
        st.markdown("## 7. Interpret the results")

        if final_mse < initial_mse:

            st.success(
                "The AI model improved image alignment "
                "for this test example."
            )

        else:

            st.warning(
                "The AI model did not improve alignment "
                "for this test example. Limited training data "
                "or insufficient training may contribute."
            )

        st.markdown("""
        ### Questions for students

        1. What information is provided to the U-Net?
        2. What does the network predict?
        3. Why do we need known displacement fields for supervised training?
        4. What is the difference between training loss and validation loss?
        5. Why do we test the model on an unseen image?
        6. What happens if test anatomy differs from training examples?
        7. How could we improve the training dataset?
        """)

        st.info(
            "The network learns from a small synthetic dataset with smooth "
            "Gaussian deformation fields. Its performance does not establish "
            "clinical accuracy. Real radiotherapy applications require "
            "realistic datasets, deformation regularization, anatomical "
            "validation, and independent clinical evaluation.",
            title="Scientific limitations",
        )

    else:

        st.info(
            "Configure the experiment and click "
            "'Generate dataset and train U-Net' to begin."
        )




# ============================================================
# 2. ORGAN SEGMENTATION
# ============================================================

elif page == "2 · AI segmentation":

    section_header(
        "Laboratory 2.2: AI-Based Organ Segmentation",
        "Generate synthetic CT and contour examples, train a U-Net, "
        "and evaluate its predictions on unseen images."
    )

    st.markdown("### Learning objectives")
    st.markdown("""
    - Generate image and reference-mask pairs for supervised learning.
    - Choose training parameters and train a small U-Net.
    - Compare training and validation losses.
    - Evaluate predictions on synthetic cases held out from training.
    - Experiment with how target organ size affects model performance.
    """)

    st.info(
        "This model learns from transformed versions of one synthetic "
        "anatomy. Results demonstrate the workflow and are not clinically "
        "validated."
    )

    st.markdown("---")
    st.markdown("## 1. Generate the synthetic dataset")

    data_col1, data_col2 = st.columns(2)
    with data_col1:
        sample_count = st.selectbox(
            "Training examples",
            [40, 80, 120, 200],
            index=1,
            key="ai_segmentation_sample_count"
        )
    with data_col2:
        organ_size_scales = {
            "Typical organ (100%)": 1.0,
            "Small organ (30%)": 0.3
        }
        selected_organ_size = st.selectbox(
            "Target organ size",
            list(organ_size_scales),
            key="ai_segmentation_training_organ_size"
        )
    selected_organ_scale = organ_size_scales[selected_organ_size]

    if st.button(
        "Generate synthetic dataset",
        type="primary",
        use_container_width=True,
        key="generate_ai_segmentation_data"
    ):
        with st.spinner("Generating CT images and reference masks..."):
            train_images, train_masks = generate_segmentation_dataset(
                sample_count,
                seed=42,
                organ_scale=selected_organ_scale
            )
            validation_images, validation_masks = (
                generate_segmentation_dataset(
                    max(20, sample_count // 4),
                    seed=123,
                    organ_scale=selected_organ_scale
                )
            )
            test_images, test_masks = generate_segmentation_dataset(
                max(20, sample_count // 4),
                seed=999,
                organ_scale=selected_organ_scale
            )

        st.session_state["ai_segmentation_data"] = {
            "train_images": train_images,
            "train_masks": train_masks,
            "validation_images": validation_images,
            "validation_masks": validation_masks,
            "test_images": test_images,
            "test_masks": test_masks,
            "organ_size": selected_organ_size,
            "organ_scale": selected_organ_scale
        }
        st.session_state.pop("ai_segmentation_results", None)
        st.session_state["ai_segmentation_evaluate_test"] = False

    if "ai_segmentation_data" in st.session_state:
        data = st.session_state["ai_segmentation_data"]

        st.success(
            f"Generated {len(data['train_images'])} training, "
            f"{len(data['validation_images'])} validation, and "
            f"{len(data['test_images'])} test image-mask pairs for "
            f"{data['organ_size'].lower()}."
        )

        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            plot_image(data["train_images"][0], "Example synthetic CT")
        with preview_col2:
            plot_image(data["train_masks"][0], "Matching reference mask")

        st.markdown("---")
        st.markdown("## 2. Choose training parameters")
        param_col1, param_col2, param_col3 = st.columns(3)

        with param_col1:
            epochs = st.selectbox(
                "Training epochs",
                [5, 10, 20, 30],
                index=1,
                key="ai_segmentation_epochs"
            )

        with param_col2:
            learning_rate = st.selectbox(
                "Learning rate",
                [0.001, 0.0005, 0.0001],
                index=1,
                key="ai_segmentation_learning_rate"
            )

        with param_col3:
            batch_size = st.selectbox(
                "Batch size",
                [4, 8, 16],
                index=1,
                key="ai_segmentation_batch_size"
            )

        st.write(f"**Computing device:** `{DEVICE}`")

        if st.button(
            "Train segmentation U-Net",
            type="primary",
            use_container_width=True,
            key="train_ai_segmentation_model"
        ):
            torch.manual_seed(42)
            np.random.seed(42)

            train_dataset = TensorDataset(
                torch.from_numpy(data["train_images"][:, None]),
                torch.from_numpy(data["train_masks"][:, None])
            )
            validation_dataset = TensorDataset(
                torch.from_numpy(data["validation_images"][:, None]),
                torch.from_numpy(data["validation_masks"][:, None])
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False
            )

            model = SegmentationUNet().to(DEVICE)
            st.markdown("## 3. Train the model")
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Training the segmentation U-Net..."):
                model, history = train_segmentation_model(
                    model,
                    train_loader,
                    validation_loader,
                    epochs,
                    learning_rate,
                    progress_bar,
                    status_text
                )

            test_tensor = torch.from_numpy(
                data["test_images"][:, None]
            ).to(DEVICE)
            model.eval()
            with torch.no_grad():
                test_probabilities = torch.sigmoid(model(test_tensor))
            predictions = (
                test_probabilities[:, 0].cpu().numpy() >= 0.5
            ).astype(np.uint8)

            test_dice = [
                dice(prediction, target)
                for prediction, target in zip(
                    predictions,
                    data["test_masks"]
                )
            ]
            test_hausdorff = [
                hausdorff_distance(prediction, target)
                for prediction, target in zip(
                    predictions,
                    data["test_masks"]
                )
            ]

            st.session_state["ai_segmentation_model"] = model
            st.session_state["ai_segmentation_results"] = {
                "history": history,
                "predictions": predictions,
                "mean_dice": float(np.mean(test_dice)),
                "mean_hausdorff": float(np.mean(test_hausdorff)),
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "organ_size": data["organ_size"]
            }

            size_summary = st.session_state.get(
                "ai_segmentation_size_summary",
                {}
            )
            size_summary[data["organ_size"]] = {
                "mean_dice": float(np.mean(test_dice)),
                "mean_hausdorff": float(np.mean(test_hausdorff))
            }
            st.session_state[
                "ai_segmentation_size_summary"
            ] = size_summary

        if "ai_segmentation_results" in st.session_state:
            results = st.session_state["ai_segmentation_results"]
            st.markdown("### Training and validation loss")
            loss_figure = go.Figure()
            loss_figure.add_trace(go.Scatter(
                y=results["history"]["training"],
                mode="lines+markers",
                name="Training loss"
            ))
            loss_figure.add_trace(go.Scatter(
                y=results["history"]["validation"],
                mode="lines+markers",
                name="Validation loss"
            ))
            loss_figure.update_layout(
                xaxis_title="Epoch",
                yaxis_title="BCE + Dice loss",
                height=380
            )
            st.plotly_chart(loss_figure, use_container_width=True)

            st.markdown("---")
            st.markdown("## 3. Evaluate on unseen test images")
            st.checkbox(
                "Calculate evaluation metrics",
                value=False,
                key="ai_segmentation_evaluate_test"
            )

        if (
            "ai_segmentation_results" in st.session_state
            and st.session_state.get("ai_segmentation_evaluate_test", False)
        ):
            results = st.session_state["ai_segmentation_results"]

            test_index = 0
            display_col1, display_col2, display_col3, display_col4 = (
                st.columns(4)
            )
            with display_col1:
                plot_image(
                    data["test_images"][test_index],
                    "Unseen test CT"
                )
            with display_col2:
                plot_image(
                    data["test_masks"][test_index],
                    "Reference mask"
                )
            with display_col3:
                plot_image(
                    results["predictions"][test_index],
                    "Predicted mask"
                )
            with display_col4:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(data["test_images"][test_index], cmap="gray")
                ax.contour(
                    results["predictions"][test_index],
                    levels=[0.5],
                    colors=["red"],
                    linewidths=1.5
                )
                ax.contour(
                    data["test_masks"][test_index],
                    levels=[0.5],
                    colors=["cyan"],
                    linewidths=1.5
                )
                ax.set_title("Overlayed masks", fontsize=11)
                ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption("Prediction: red | Reference: cyan")

            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.metric(
                    "Mean test Dice",
                    f"{results['mean_dice']:.4f}"
                )
            with result_col2:
                hausdorff_value = results["mean_hausdorff"]
                hausdorff_text = (
                    "Inf" if np.isinf(hausdorff_value)
                    else f"{hausdorff_value:.2f} px"
                )
                st.metric("Mean test Hausdorff distance", hausdorff_text)

            st.caption(
                f"Test metrics average {len(data['test_images'])} "
                "held-out examples."
            )

        if (
            "ai_segmentation_results" in st.session_state
            and st.session_state.get("ai_segmentation_evaluate_test", False)
        ):
            st.markdown("---")
            st.markdown("## 4. Compare typical and small-organ runs")
            st.write(
                "To run the small-organ experiment, select **Small organ "
                "(30%)** above, generate the dataset, then train and evaluate "
                "that model using the same steps. Repeat with the typical "
                "organ size to compare performance."
            )

            size_summary = st.session_state.get(
                "ai_segmentation_size_summary",
                {}
            )
            if size_summary:
                summary_order = [
                    label for label in organ_size_scales
                    if label in size_summary
                ]
                comparison_figure = go.Figure(go.Bar(
                    x=summary_order,
                    y=[
                        size_summary[label]["mean_dice"]
                        for label in summary_order
                    ],
                    name="Mean test Dice"
                ))
                comparison_figure.update_layout(
                    title="Held-out Dice after training at each organ size",
                    xaxis_title="Training and test organ size",
                    yaxis_title="Mean test Dice",
                    yaxis_range=[0, 1],
                    height=360
                )
                st.plotly_chart(
                    comparison_figure,
                    use_container_width=True
                )

                for size_label in summary_order:
                    summary = size_summary[size_label]
                    hd_value = summary["mean_hausdorff"]
                    hd_text = (
                        "Inf" if np.isinf(hd_value)
                        else f"{hd_value:.2f} px"
                    )
                    st.write(
                        f"**{size_label}:** mean test Dice "
                        f"{summary['mean_dice']:.4f}; mean test "
                        f"Hausdorff distance {hd_text}."
                    )

            st.markdown("### Questions for students")
            st.markdown("""
            1. How do training and validation losses change as epochs increase?
            2. How well does the predicted contour match the reference?
            3. What does the Dice score tell you that Hausdorff distance does not?
            4. How could more varied training data improve generalization?
            """)

    else:
        st.info("Generate the dataset to continue to model training.")

elif page == "2 · Organ segmentation":

    section_header(
        "Laboratory 2: Manual Organ-at-Risk Segmentation",
        "Draw a polygon around the organ, generate a binary mask, "
        "and compare your contour with the reference segmentation."
    )

    st.markdown("### Introduction")

    st.write("""
    In radiotherapy, organs at risk (OARs) are delineated on medical
    images to support treatment planning and dose evaluation.

    In this exercise, you will manually contour a synthetic anatomical
    structure using polygon drawing. The application will convert your
    contour into a binary mask and calculate its overlap with a hidden
    reference mask.
    """)

    # --------------------------------------------------------
    # 1. PREPARE THE SYNTHETIC CT IMAGE AND REFERENCE MASK
    # --------------------------------------------------------

    fixed_image = make_ct_phantom()
    reference_mask = make_segmentation_mask()

    image_height, image_width = fixed_image.shape

    if reference_mask.shape != fixed_image.shape:
        st.error(
            "The CT image and reference mask must have identical dimensions."
        )
        st.stop()

    # Ensure reference mask is binary.
    reference_mask = (reference_mask > 0).astype(np.uint8)

    # Canvas dimensions.
    canvas_size = 512

    # Prepare CT image.
    image_uint8 = (
        np.clip(fixed_image, 0, 1) * 255
    ).astype(np.uint8)

    ct_pil = Image.fromarray(image_uint8).convert("RGB")

    ct_canvas_background = ct_pil.resize(
        (canvas_size, canvas_size),
        Image.Resampling.BILINEAR
    )

    # Prepare reference mask in canvas coordinates.
    reference_mask_canvas = Image.fromarray(
        (reference_mask * 255).astype(np.uint8)
    ).resize(
        (canvas_size, canvas_size),
        Image.Resampling.NEAREST
    )

    reference_mask_canvas = (
        np.asarray(reference_mask_canvas) > 0
    ).astype(np.uint8)

    st.markdown("---")
    st.markdown("## Step 1: Inspect the CT image")

    st.write(
        "The reference contour is hidden initially. Identify the "
        "anatomical structure that you need to delineate."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            ct_pil,
            caption="Synthetic CT image",
            use_container_width=True
        )

    with col2:
        st.markdown("**Segmentation instructions**")
        st.markdown("""
        1. Select the polygon tool on the drawing canvas.
        2. Click successive points along the organ boundary.
        3. Double-click the final point to close the polygon.
        4. Inspect the filled polygon.
        5. Click **Generate segmentation mask**.
        """)

    # --------------------------------------------------------
    # 2. MANUAL POLYGON DRAWING
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## Step 2: Draw the organ contour")

    if "segmentation_canvas_reset" not in st.session_state:
        st.session_state["segmentation_canvas_reset"] = 0

    reset_col, info_col = st.columns([1, 3])

    with reset_col:
        clear_drawing = st.button(
            "Clear drawing",
            key="clear_manual_segmentation_drawing"
        )

    if clear_drawing:
        st.session_state["segmentation_canvas_reset"] += 1
        st.session_state.pop("student_segmentation_mask", None)
        st.rerun()

    with info_col:
        st.caption(
            "Click around the organ boundary. Double-click to finish "
            "and close the polygon."
        )

    canvas_result = st_canvas(
        fill_color="rgba(255, 40, 40, 0.30)",
        stroke_width=3,
        stroke_color="#FF2828",
        background_image=ct_canvas_background,
        height=canvas_size,
        width=canvas_size,
        drawing_mode="polygon",
        update_streamlit=True,
        key=(
            f"segmentation_canvas_"
            f"{st.session_state['segmentation_canvas_reset']}"
        ),
        return_image_data=True,
    )

    # --------------------------------------------------------
    # 3. CONVERT THE DRAWN POLYGON INTO A BINARY MASK
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## Step 3: Generate the binary mask")

    if st.button(
        "Generate segmentation mask",
        type="primary",
        use_container_width=True,
        key="generate_manual_segmentation_mask"
    ):

        canvas_json = canvas_result.json_data
        canvas_image = canvas_result.image_data

        objects = (
            canvas_json.get("objects", [])
            if isinstance(canvas_json, dict)
            else []
        )

        has_polygon = any(
            str(obj.get("type", "")).lower() == "polygon"
            for obj in objects
        )

        if not has_polygon or canvas_image is None:

            st.warning(
                "No completed polygon was detected. Draw the contour "
                "and double-click the final point to close it."
            )

        else:

            # Detect red polygon pixels in the canvas image.
            red_pixels = (
                (canvas_image[:, :, 0] >
                 canvas_image[:, :, 1] + 15)
                &
                (canvas_image[:, :, 0] >
                 canvas_image[:, :, 2] + 15)
            )

            mask_canvas = (
                red_pixels.astype(np.uint8) * 255
            )

            # Convert canvas mask to PIL image.
            mask_pil = Image.fromarray(mask_canvas)

            # Resize to original CT dimensions.
            mask_original = mask_pil.resize(
                (image_width, image_height),
                Image.Resampling.NEAREST
            )

            student_mask = (
                np.asarray(mask_original) > 0
            ).astype(np.uint8)

            if student_mask.sum() < 5:

                st.warning(
                    "The selected polygon is too small or could not "
                    "be converted. Please redraw the contour."
                )

            else:

                st.session_state[
                    "student_segmentation_mask"
                ] = student_mask

                st.success(
                    "The manual segmentation mask has been generated."
                )

    # --------------------------------------------------------
    # 4. DISPLAY THE STUDENT MASK AND ITS OVERLAY
    # --------------------------------------------------------

    student_mask = st.session_state.get(
        "student_segmentation_mask"
    )

    if student_mask is not None:

        # Check dimensions before visualization and evaluation.
        if student_mask.shape != fixed_image.shape:
            st.error(
                "The student mask dimensions do not match the CT image."
            )
            st.stop()

        st.markdown("---")
        st.markdown("## Step 4: Visualize your segmentation")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Reference mask**")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(
                reference_mask,
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            st.markdown("**Manual binary mask**")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(
                student_mask,
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col3:
            st.markdown("**Both masks overlaid on CT**")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(fixed_image, cmap="gray", vmin=0, vmax=1)
            ax.contour(
                student_mask,
                levels=[0.5],
                colors=["red"],
                linewidths=1.8
            )
            ax.contour(
                reference_mask,
                levels=[0.5],
                colors=["cyan"],
                linewidths=1.8
            )
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Manual contour: red | Reference contour: cyan")

        st.write(
            f"Number of pixels in the student mask: "
            f"{int(student_mask.sum())}"
        )
        st.write(
            f"Number of pixels in the reference mask: "
            f"{int(reference_mask.sum())}"
        )

        # ----------------------------------------------------
        # 5. CALCULATE THE COMPARISON METRICS
        # ----------------------------------------------------

        st.markdown("---")
        st.markdown("## Step 5: Evaluate your segmentation")

        calculate_dice = st.checkbox(
            "Calculate evaluation metrics",
            value=False,
            key="reveal_segmentation_reference"
        )

        if calculate_dice:

            # Verify that both masks have identical dimensions.
            if student_mask.shape != reference_mask.shape:
                st.error(
                    "Student and reference masks have different dimensions."
                )
                st.stop()

            pixel_spacing_mm = st.number_input(
                "Pixel spacing (mm/pixel)",
                min_value=0.01,
                value=1.0,
                step=0.1,
                help=(
                    "The synthetic CT has no physical scale. Replace the "
                    "illustrative default with the image's known pixel "
                    "spacing to get a meaningful distance in millimeters."
                )
            )

            # ------------------------------------------------
            # DICE EVALUATION
            # ------------------------------------------------

            segmentation_dice = dice(
                student_mask,
                reference_mask
            )
            segmentation_hausdorff = hausdorff_distance(
                student_mask,
                reference_mask
            )
            hausdorff_mm = segmentation_hausdorff * pixel_spacing_mm

            st.markdown("### Quantitative evaluation")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    "Dice Similarity Coefficient",
                    f"{segmentation_dice:.4f}"
                )

            with metric_col2:
                hausdorff_display = (
                    "Inf" if np.isinf(segmentation_hausdorff)
                    else f"{segmentation_hausdorff:.2f} px"
                )
                st.metric(
                    "Hausdorff distance (px)",
                    hausdorff_display
                )

            with metric_col3:
                hausdorff_mm_display = (
                    "Inf" if np.isinf(hausdorff_mm)
                    else f"{hausdorff_mm:.2f} mm"
                )
                st.metric(
                    "Hausdorff distance (mm)",
                    hausdorff_mm_display
                )

            st.progress(
                float(np.clip(segmentation_dice, 0, 1))
            )

            st.write("""
            The Dice Similarity Coefficient measures the spatial
            overlap between the student mask and the reference mask.

            A value of 1 indicates perfect overlap, while a value
            of 0 indicates no overlap.

            The Hausdorff distance is the largest distance from a
            boundary point in either mask to the nearest boundary
            point in the other mask. It is reported in pixels and
            emphasizes the largest contour mismatch.

            The millimeter value uses the pixel spacing entered above.
            """)

        else:

            st.info(
                "Select 'Calculate evaluation metrics' to view the quantitative "
                "overlap metrics."
            )

    else:

        st.info(
            "Draw and close a polygon, then click "
            "'Generate segmentation mask' to continue."
        )

    # --------------------------------------------------------
    # 6. EDUCATIONAL QUESTIONS
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### Questions for students")

    st.markdown("""
    1. What is an organ at risk?
    2. Why is accurate manual contouring important in radiotherapy?
    3. What does the binary mask represent?
    4. What does the Dice coefficient measure?
    5. What could explain a low Dice score?
    6. Can a high Dice score guarantee a clinically acceptable contour?
    7. Why should segmentation be evaluated using more than one metric?
    """)

    st.info(
        "Educational demonstration using synthetic images only. "
        "The reference mask is a predefined synthetic contour and "
        "does not represent a clinically validated organ delineation."
    )



# ============================================================
# 3. DOSE PREDICTION
# ============================================================

elif page == "3 · Dose prediction":

    section_header(
        "Laboratory 3.1: Beam Planning Experience",
        "Explore the concept of dose distribution prediction."
    )

    st.markdown("### Learning objectives")
    st.markdown("""
    - Identify the segmented region on a CT slice.
    - Choose two beam orientations and adjust their field width.
    - Observe how the two-beam arrangement changes target coverage.
    - Compare a simplified 3D dose distribution with a reference example.
    """)

    st.markdown("## 1. Choose and adapt two beams")
    st.write(
        "Inspect the CT slice and segmented region. Choose two beam "
        "orientations and adjust each field width to cover it."
    )

    beam_ct_slice = make_ct_phantom()
    beam_target_mask = make_segmentation_mask()
    (
        beam_control_col1,
        beam_control_col2,
        beam_control_col3,
        beam_control_col4
    ) = st.columns(4)

    with beam_control_col1:
        beam_orientation_1 = st.slider(
            "Beam 1 orientation (degrees)",
            min_value=0,
            max_value=355,
            value=90,
            step=5,
            help="0° enters from the left; 90° enters from the top."
        )

    with beam_control_col2:
        beam_field_width_1_px = st.slider(
            "Beam 1 field width (pixels)",
            min_value=8,
            max_value=64,
            value=40,
            step=2
        )

    with beam_control_col3:
        beam_orientation_2 = st.slider(
            "Beam 2 orientation (degrees)",
            min_value=0,
            max_value=355,
            value=0,
            step=5,
            help="0° enters from the left; 90° enters from the top."
        )

    with beam_control_col4:
        beam_field_width_2_px = st.slider(
            "Beam 2 field width (pixels)",
            min_value=8,
            max_value=64,
            value=56,
            step=2,
            key="dose_beam_2_width"
        )

    beam_field_1 = make_2d_beam_field(
        beam_target_mask,
        beam_orientation_1,
        beam_field_width_1_px
    )
    beam_field_2 = make_2d_beam_field(
        beam_target_mask,
        beam_orientation_2,
        beam_field_width_2_px
    )
    combined_beam_field = beam_field_1 | beam_field_2
    target_pixel_count = int(beam_target_mask.sum())
    target_coverage = (
        float(np.logical_and(combined_beam_field, beam_target_mask).sum())
        / max(target_pixel_count, 1)
        * 100
    )

    beam_fig, beam_ax = plt.subplots(figsize=(6, 5))
    beam_ax.imshow(beam_ct_slice, cmap="gray", vmin=0, vmax=1)
    beam_ax.imshow(
        np.ma.masked_where(~beam_field_1, beam_field_1),
        cmap="autumn",
        alpha=0.30,
        vmin=0,
        vmax=1
    )
    beam_ax.imshow(
        np.ma.masked_where(~beam_field_2, beam_field_2),
        cmap="winter",
        alpha=0.25,
        vmin=0,
        vmax=1
    )
    beam_ax.contour(
        beam_target_mask,
        levels=[0.5],
        colors=["lime"],
        linewidths=2
    )
    beam_ax.set_title("CT slice with two beam fields and target")
    beam_ax.axis("off")
    plt.tight_layout()
    st.pyplot(beam_fig, use_container_width=True)
    plt.close(beam_fig)

    coverage_col1, coverage_col2 = st.columns(2)
    with coverage_col1:
        st.metric("Segmented-region coverage", f"{target_coverage:.1f}%")
    with coverage_col2:
        st.caption(
            "Beam 1: orange | Beam 2: blue | Target: green. "
            "Adjust each orientation and field width."
        )

    if target_coverage >= 99:
        st.success("The two-beam fields cover the full segmented region.")
    else:
        st.warning(
            "Part of the segmented region is outside both beam fields. "
            "Try changing their orientations or increasing the field width."
        )

    st.caption(
        "Simplified 2D geometry for teaching. It is not a treatment plan "
        "or a physical dose calculation."
    )

    st.markdown("## 2. Compare with the reference distribution")
    st.markdown("### Simulation parameters")

    beam_energy = st.selectbox(
        "Illustrative beam energy (MV)",
        [6, 10, 15],
        index=2
    )
    st.write(
        f"Dose uses both orientations and field widths from Step 1 "
        f"({beam_field_width_1_px} px and {beam_field_width_2_px} px). "
        "Widening either field can "
        "increase target coverage and non-target V95."
    )

    size = 64
    dose_target_mask = make_segmentation_mask(size=size).astype(bool)
    target_y_indices, target_x_indices = np.nonzero(dose_target_mask)
    default_reference_x = int(round(target_x_indices.mean()))
    default_reference_y = int(round(target_y_indices.mean()))

    st.markdown("**Maximum-dose reference point on the CT slice**")
    point_col1, point_col2 = st.columns(2)
    with point_col1:
        dose_reference_x = st.slider(
            "Reference point X (pixels)",
            min_value=0,
            max_value=size - 1,
            value=default_reference_x,
            key="dose_reference_point_x"
        )
    with point_col2:
        dose_reference_y = st.slider(
            "Reference point Y (pixels)",
            min_value=0,
            max_value=size - 1,
            value=default_reference_y,
            key="dose_reference_point_y"
        )
    reference_point_xy = (dose_reference_x, dose_reference_y)
    st.caption(
        "These coordinates move the dose peak within the axial slice; "
        "the depth profile remains centered near slice 36."
    )

    dose = make_synthetic_dose_distribution(
        beam_energy,
        beam_field_widths_px=(
            beam_field_width_1_px,
            beam_field_width_2_px
        ),
        beam_orientations=(beam_orientation_1, beam_orientation_2),
        reference_point_xy=reference_point_xy,
        size=size
    )

    peak_dose_slice = int(
        np.argmin(np.abs(np.linspace(-1, 1, size) - 0.15))
    )
    slice_index = st.slider(
        "Axial slice index",
        min_value=0,
        max_value=size - 1,
        value=peak_dose_slice
    )

    reference_dose = make_ideal_reference_dose_distribution(size=size)
    student_slice = dose[slice_index, :, :]
    reference_slice = reference_dose[slice_index, :, :]
    student_isodose_slice = student_slice
    reference_isodose_slice = reference_slice
    target_volume = np.broadcast_to(
        dose_target_mask,
        dose.shape
    )
    student_target_dose = dose[target_volume]
    reference_target_dose = reference_dose[target_volume]
    student_non_target_dose = dose[~target_volume]
    student_d95 = float(np.percentile(student_target_dose, 5))
    reference_d95 = float(np.percentile(reference_target_dose, 5))
    student_v95 = float(np.mean(student_target_dose >= 95) * 100)
    reference_v95 = float(np.mean(reference_target_dose >= 95) * 100)
    student_non_target_v95 = float(
        np.mean(student_non_target_dose >= 95) * 100
    )

    st.write(
        "The reference is an ideal, beam-independent dose map shaped to "
        "the segmented target. Its 95% isodose covers approximately 95% "
        f"of the target (calculated V95: {reference_v95:.1f}%). D95 is "
        "calculated from the same reference map. The student dose is still "
        "generated from the two beam settings selected in Step 1."
    )

    dose_ct_slice = make_ct_phantom(size=size)
    isodose_levels = [10, 20, 30, 95]
    student_isodose_levels = isodose_levels
    reference_isodose_levels = isodose_levels
    isodose_colors = ["cyan", "yellow", "red", "magenta"]

    compare_col1, compare_col2 = st.columns(2)
    with compare_col1:
        compare_fig, compare_ax = plt.subplots(figsize=(4, 4))
        compare_ax.imshow(dose_ct_slice, cmap="gray", vmin=0, vmax=1)
        student_contours = compare_ax.contour(
            student_isodose_slice,
            levels=student_isodose_levels,
            colors=isodose_colors,
            linewidths=1.5
        )
        compare_ax.clabel(
            student_contours,
            fmt="%.0f%%",
            fontsize=7,
            inline=True
        )
        compare_ax.contour(
            dose_target_mask,
            levels=[0.5],
            colors=["lime"],
            linewidths=1.2
        )
        compare_ax.scatter(
            [dose_reference_x],
            [dose_reference_y],
            marker="x",
            color="white",
            s=70,
            linewidths=2
        )
        compare_ax.set_title("Student isodoses on CT")
        compare_ax.axis("off")
        plt.tight_layout()
        st.pyplot(compare_fig, use_container_width=True)
        plt.close(compare_fig)
        st.caption(
            "10% cyan | 20% yellow | 30% red | 95% magenta | "
            "Target: green | Max-dose point: white x"
        )
        if float(student_slice.max()) < 95:
            st.caption(
                "No student 95% isodose appears on this slice because its "
                f"maximum dose is {student_slice.max():.1f}%. Increase the "
                "beam energy or select a higher-dose slice."
            )

    with compare_col2:
        compare_fig, compare_ax = plt.subplots(figsize=(4, 4))
        compare_ax.imshow(dose_ct_slice, cmap="gray", vmin=0, vmax=1)
        reference_contours = compare_ax.contour(
            reference_isodose_slice,
            levels=reference_isodose_levels,
            colors=isodose_colors,
            linewidths=1.5
        )
        compare_ax.clabel(
            reference_contours,
            fmt="%.0f%%",
            fontsize=7,
            inline=True
        )
        compare_ax.contour(
            dose_target_mask,
            levels=[0.5],
            colors=["lime"],
            linewidths=1.2
        )
        compare_ax.set_title("Ideal reference isodoses on CT")
        compare_ax.axis("off")
        plt.tight_layout()
        st.pyplot(compare_fig, use_container_width=True)
        plt.close(compare_fig)
        st.caption(
            "Prescription dose: 10% cyan | 20% yellow | 30% red | "
            "95% magenta | Target: green"
        )

    (
        coverage_col1,
        coverage_col2,
        coverage_col3,
        coverage_col4,
        coverage_col5
    ) = st.columns(5)
    with coverage_col1:
        st.metric("Student target V95", f"{student_v95:.1f}%")
    with coverage_col2:
        st.metric("Reference target V95", f"{reference_v95:.1f}%")
    with coverage_col3:
        st.metric("Student target D95", f"{student_d95:.1f}%")
    with coverage_col4:
        st.metric("Reference target D95", f"{reference_d95:.1f}%")
    with coverage_col5:
        st.metric(
            "Non-target V95",
            f"{student_non_target_v95:.1f}%",
            help=(
                "Percent of non-target synthetic volume receiving at "
                "least 95% of prescription dose."
            )
        )

    st.caption(
        "The reference is a target-shaped teaching benchmark, not a beam "
        "calculation. Its 95% isodose and coverage metrics are generated "
        "directly from the segmented target. D95 is reported separately."
    )

    if student_v95 >= 95 and student_d95 >= 95:
        st.success(
            "The dose meets the 95/95 target-coverage criterion."
        )
    else:
        st.warning(
            "The dose is below the 95/95 target-coverage criterion. "
            "Adjust beam orientations, field widths, energy, "
            "or reference point."
        )

    dose_thresholds = np.linspace(0, 100, 101)
    student_vx = [
        float(np.mean(student_target_dose >= threshold) * 100)
        for threshold in dose_thresholds
    ]
    reference_vx = [
        float(np.mean(reference_target_dose >= threshold) * 100)
        for threshold in dose_thresholds
    ]
    dvh_figure = go.Figure()
    dvh_figure.add_trace(go.Scatter(
        x=dose_thresholds,
        y=student_vx,
        mode="lines",
        line_shape="hv",
        name="Student dose"
    ))
    dvh_figure.add_trace(go.Scatter(
        x=dose_thresholds,
        y=reference_vx,
        mode="lines",
        line_shape="hv",
        name="Reference dose"
    ))
    dvh_figure.add_vline(
        x=95,
        line_dash="dash",
        line_color="gray"
    )
    dvh_figure.add_hline(
        y=95,
        line_dash="dash",
        line_color="gray"
    )
    dvh_figure.update_layout(
        title="Target cumulative dose-volume histogram",
        xaxis_title="Dose (% of prescription)",
        yaxis_title="Target volume receiving at least that dose (%)",
        xaxis_range=[0, 100],
        yaxis_range=[0, 100],
        height=420
    )
    st.plotly_chart(dvh_figure, use_container_width=True)
    st.caption(
        "Vx is the percentage of target volume receiving at least x% of the "
        "prescription dose. The dashed lines mark the 95% dose and 95% volume "
        "coverage criterion. The reference target is extruded across the "
        "synthetic depth."
    )

    st.write("""
    The displayed distribution is a simplified mathematical
    illustration. It does not represent a Monte Carlo calculation,
    treatment planning system calculation or trained AI prediction.
    """)

    st.markdown("### Questions for students")

    st.markdown("""
    1. What is a 3D dose distribution?
    2. What information could be used as input to a dose prediction model?
    3. Why is a 3D model potentially more informative than a 2D model?
    4. Which metrics could evaluate dose prediction?
    5. Why must a predicted dose distribution be independently validated?
    """)


# ============================================================
# AI-BASED DOSE PREDICTION
# ============================================================

elif page.startswith("3 ") and page.endswith("AI dose prediction"):

    section_header(
        "Laboratory 3.2: AI-Based Dose Prediction",
        "Train a model to recommend beam energy, orientations, and field widths for an ideal target coverage."
    )

    st.markdown("### Learning objectives")
    st.markdown("""
    - Generate target cases and beam plans that meet the ideal 95% coverage goal.
    - Train a model to recommend energy and settings for two beams.
    - Evaluate the recommended plan against the ideal target-shaped dose reference.
    """)

    st.markdown("## 1. Generate training data")
    st.write(
        "The model receives target size, position, and orientation as inputs. "
        "Training labels are two-beam plans aligned with the target axes. "
        "Field widths are adjusted until at least 95% of target volume "
        "receives 95% of the ideal reference dose."
    )
    training_case_count = st.slider(
        "Number of synthetic target cases",
        min_value=12,
        max_value=80,
        value=32,
        step=4,
        key="ai_dose_plan_case_count"
    )

    if st.button("Generate beam-planning cases", type="primary"):
        rng = np.random.default_rng(203)
        example_cases = [
            make_dose_plan_training_case(
                rng, scale=0.65, rotation=15.0, shift_x=-0.03, shift_y=0.02
            ),
            make_dose_plan_training_case(
                rng, scale=0.90, rotation=55.0, shift_x=0.02, shift_y=-0.01
            ),
            make_dose_plan_training_case(
                rng, scale=1.15, rotation=135.0, shift_x=-0.01, shift_y=-0.02
            )
        ]
        training_cases = example_cases + [
            make_dose_plan_training_case(rng)
            for _ in range(training_case_count - len(example_cases))
        ]
        validation_cases = [
            make_dose_plan_training_case(rng)
            for _ in range(max(4, training_case_count // 4))
        ]
        test_case = make_dose_plan_training_case(
            rng, scale=0.91, rotation=37.0, shift_x=0.02, shift_y=-0.02
        )
        st.session_state["ai_dose_plan_data"] = {
            "train_x": np.stack([case[0] for case in training_cases]),
            "train_y": np.stack([case[1] for case in training_cases]),
            "train_coverages": [case[6] for case in training_cases],
            "validation_x": np.stack([case[0] for case in validation_cases]),
            "validation_y": np.stack([case[1] for case in validation_cases]),
            "test_case": test_case,
            "example_cases": example_cases
        }
        st.session_state.pop("ai_dose_plan_model", None)
        st.session_state.pop("ai_dose_plan_history", None)
        st.session_state.pop("ai_dose_plan_result", None)

    if "ai_dose_plan_data" in st.session_state:
        plan_data = st.session_state["ai_dose_plan_data"]
        st.success(
            f"Generated {len(plan_data['train_x'])} training cases and "
            f"{len(plan_data['validation_x'])} validation cases. Every "
            "reference beam plan achieves at least 95% target V95."
        )
        st.caption(
            "Training-plan target V95 range: "
            f"{min(plan_data['train_coverages']):.1f} to "
            f"{max(plan_data['train_coverages']):.1f}%."
        )
        st.caption(
            "This simulator scales dose by energy relative to 15 MV, so "
            "15 MV is required to meet the 95% dose coverage goal."
        )

        example_slice = int(np.argmin(np.abs(np.linspace(-1, 1, 32) - 0.15)))
        example_columns = st.columns(len(plan_data["example_cases"]))
        example_labels = ["Small target", "Typical target", "Large target"]
        example_scales = [65, 90, 115]
        for example_column, example_case, example_label, example_scale in zip(
            example_columns,
            plan_data["example_cases"],
            example_labels,
            example_scales
        ):
            example_mask = example_case[2]
            example_reference = example_case[3]
            example_plan_dose = example_case[4]
            example_plan = example_case[5]
            example_rotation = float(
                np.degrees(np.arctan2(example_case[0][3], example_case[0][4]))
            )
            example_ct = make_ct_phantom(
                size=32,
                target_mask=example_mask
            )
            beam_field_1 = make_2d_beam_field(
                example_mask,
                example_plan[1],
                example_plan[3]
            )
            beam_field_2 = make_2d_beam_field(
                example_mask,
                example_plan[2],
                example_plan[4]
            )
            with example_column:
                example_fig, example_ax = plt.subplots(figsize=(4, 4))
                example_ax.imshow(example_ct, cmap="gray", vmin=0, vmax=1)
                example_ax.imshow(
                    np.ma.masked_where(~beam_field_1, beam_field_1),
                    cmap="autumn",
                    alpha=0.12,
                    vmin=0,
                    vmax=1
                )
                example_ax.imshow(
                    np.ma.masked_where(~beam_field_2, beam_field_2),
                    cmap="winter",
                    alpha=0.12,
                    vmin=0,
                    vmax=1
                )
                example_ax.contour(
                    example_reference[example_slice],
                    levels=[95],
                    colors=["cyan"],
                    linewidths=1.8
                )
                example_ax.contour(
                    example_plan_dose[example_slice],
                    levels=[95],
                    colors=["magenta"],
                    linewidths=1.8
                )
                example_ax.contour(
                    example_mask,
                    levels=[0.5],
                    colors=["lime"],
                    linewidths=1.2
                )
                example_ax.set_title(
                    f"{example_label}: {example_scale}% size, "
                    f"{example_rotation:.0f} deg rotation"
                )
                example_ax.axis("off")
                plt.tight_layout()
                st.pyplot(example_fig, use_container_width=True)
                plt.close(example_fig)
                st.caption(
                    f"95% isodose: ideal cyan | plan magenta | target green. "
                    f"Beam 1 orange: {example_plan[1]:.0f} deg, "
                    f"{example_plan[3]} px; Beam 2 blue: "
                    f"{example_plan[2]:.0f} deg, {example_plan[4]} px; "
                    f"energy {example_plan[0]} MV; "
                    f"target V95 {example_case[6]:.1f}%."
                )

        st.markdown("**Generated beam configurations**")
        configuration_table = [
            "| Example | Target size | Rotation | Energy | Beam 1 angle / width | Beam 2 angle / width | Target V95 |",
            "|---|---:|---:|---:|---:|---:|---:|"
        ]
        for example_label, example_scale, example_case in zip(
            example_labels,
            example_scales,
            plan_data["example_cases"]
        ):
            example_plan = example_case[5]
            example_rotation = float(
                np.degrees(np.arctan2(example_case[0][3], example_case[0][4]))
            )
            configuration_table.append(
                f"| {example_label} | {example_scale}% | "
                f"{example_rotation:.0f} deg | {example_plan[0]} MV | "
                f"{example_plan[1]:.0f} deg / {example_plan[3]} px | "
                f"{example_plan[2]:.0f} deg / {example_plan[4]} px | "
                f"{example_case[6]:.1f}% |"
            )
        st.markdown("\n".join(configuration_table))

        st.markdown("## 2. Train the beam recommendation model")
        train_control_col1, train_control_col2 = st.columns(2)
        with train_control_col1:
            training_epochs = st.slider(
                "Training epochs", min_value=20, max_value=200,
                value=100, step=20, key="ai_dose_plan_epochs"
            )
        with train_control_col2:
            hidden_units = st.select_slider(
                "Model size", options=[16, 32, 64], value=32,
                key="ai_dose_plan_hidden_units"
            )

        if st.button("Train beam recommendation model", type="primary"):
            torch.manual_seed(42)
            model = BeamPlanRecommendationMLP(hidden_units=hidden_units).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_function = nn.MSELoss()
            train_x = torch.from_numpy(plan_data["train_x"]).to(DEVICE)
            train_y = torch.from_numpy(plan_data["train_y"]).to(DEVICE)
            validation_x = torch.from_numpy(plan_data["validation_x"]).to(DEVICE)
            validation_y = torch.from_numpy(plan_data["validation_y"]).to(DEVICE)
            train_history = []
            validation_history = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            with st.spinner("Training the beam recommendation model..."):
                for epoch in range(training_epochs):
                    model.train()
                    loss = loss_function(model(train_x), train_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    training_loss = float(loss.detach())
                    train_history.append(training_loss)
                    model.eval()
                    with torch.no_grad():
                        validation_loss = loss_function(model(validation_x), validation_y)
                    validation_loss_value = float(validation_loss)
                    validation_history.append(validation_loss_value)
                    progress_bar.progress((epoch + 1) / training_epochs)
                    status_text.text(
                        f"Epoch {epoch + 1}/{training_epochs} | "
                        f"Training loss: {training_loss:.5f} | "
                        f"Validation loss: {validation_loss_value:.5f}"
                    )
            status_text.text("Training complete.")
            st.session_state["ai_dose_plan_model"] = model
            st.session_state["ai_dose_plan_history"] = {
                "train": train_history, "validation": validation_history
            }
            st.session_state.pop("ai_dose_plan_result", None)

        if "ai_dose_plan_history" in st.session_state:
            history = st.session_state["ai_dose_plan_history"]
            loss_figure = go.Figure()
            loss_figure.add_trace(go.Scatter(
                x=list(range(1, len(history["train"]) + 1)),
                y=history["train"], mode="lines+markers", name="Training loss"
            ))
            loss_figure.add_trace(go.Scatter(
                x=list(range(1, len(history["validation"]) + 1)),
                y=history["validation"], mode="lines+markers", name="Validation loss"
            ))
            loss_figure.update_layout(
                title="Beam-parameter training and validation loss",
                xaxis_title="Epoch", yaxis_title="Mean squared parameter error",
                height=350
            )
            st.plotly_chart(loss_figure, use_container_width=True)

            st.markdown("## 3. Evaluate the recommendation on an unseen target")
            if st.button("Recommend beams for unseen target"):
                case = plan_data["test_case"]
                model = st.session_state["ai_dose_plan_model"]
                model.eval()
                with torch.no_grad():
                    raw = model(torch.from_numpy(case[0]).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
                energy_score = float(np.clip(raw[0], 0, 1))
                energy = min([6, 10, 15], key=lambda value: abs(value / 15 - energy_score))
                angle_1 = float(np.degrees(np.arctan2(raw[1], raw[2])) % 360)
                angle_2 = float(np.degrees(np.arctan2(raw[3], raw[4])) % 360)
                width_1 = int(np.clip(round(raw[5] * 64), 8, 64))
                width_2 = int(np.clip(round(raw[6] * 64), 8, 64))
                target_mask = case[2]
                target_y, target_x = np.nonzero(target_mask)
                center_point = (int(round(target_x.mean())), int(round(target_y.mean())))
                recommended_dose = make_synthetic_dose_distribution(
                    energy,
                    beam_field_widths_px=(width_1, width_2),
                    beam_orientations=(angle_1, angle_2),
                    reference_point_xy=center_point,
                    size=32,
                    target_mask=target_mask
                )
                target_volume = np.broadcast_to(target_mask, recommended_dose.shape)
                target_v95 = float(np.mean(recommended_dose[target_volume] >= 95) * 100)
                coverage_adjusted = False
                while target_v95 < 95 and (width_1 < 64 or width_2 < 64):
                    width_1 = min(width_1 + 2, 64)
                    width_2 = min(width_2 + 2, 64)
                    recommended_dose = make_synthetic_dose_distribution(
                        energy,
                        beam_field_widths_px=(width_1, width_2),
                        beam_orientations=(angle_1, angle_2),
                        reference_point_xy=center_point,
                        size=32,
                        target_mask=target_mask
                    )
                    target_v95 = float(
                        np.mean(recommended_dose[target_volume] >= 95) * 100
                    )
                    coverage_adjusted = True
                if target_v95 < 95:
                    energy = 15
                    width_1 = 64
                    width_2 = 64
                    recommended_dose = make_synthetic_dose_distribution(
                        energy,
                        beam_field_widths_px=(width_1, width_2),
                        beam_orientations=(angle_1, angle_2),
                        reference_point_xy=center_point,
                        size=32,
                        target_mask=target_mask
                    )
                    target_v95 = float(
                        np.mean(recommended_dose[target_volume] >= 95) * 100
                    )
                    coverage_adjusted = True
                reference_dose = case[3]
                target_mae = float(np.mean(np.abs(
                    recommended_dose[target_volume] - reference_dose[target_volume]
                )))
                st.session_state["ai_dose_plan_result"] = {
                    "energy": energy, "angle_1": angle_1, "angle_2": angle_2,
                    "width_1": width_1, "width_2": width_2,
                    "dose": recommended_dose, "v95": target_v95,
                    "mae": target_mae, "coverage_adjusted": coverage_adjusted
                }

            if "ai_dose_plan_result" in st.session_state:
                result = st.session_state["ai_dose_plan_result"]
                case = plan_data["test_case"]
                target_mask = case[2]
                reference_dose = case[3]
                recommended_dose = result["dose"]
                target_volume = np.broadcast_to(target_mask, recommended_dose.shape)
                reference_v95 = float(np.mean(reference_dose[target_volume] >= 95) * 100)
                coverage_metrics = st.columns(3)
                coverage_metrics[0].metric(
                    "Ideal reference target V95",
                    f"{reference_v95:.1f}%"
                )
                coverage_metrics[1].metric(
                    "Recommended plan target V95",
                    f"{result['v95']:.1f}%"
                )
                coverage_metrics[2].metric(
                    "Target dose MAE vs ideal",
                    f"{result['mae']:.2f}%"
                )
                if result["coverage_adjusted"]:
                    st.caption(
                        "Beam widths were automatically increased to meet "
                        "the 95% target V95 requirement."
                    )

                slice_index = int(np.argmin(np.abs(np.linspace(-1, 1, 32) - 0.15)))
                ct_slice = make_ct_phantom(
                    size=32,
                    target_mask=target_mask
                )
                panels = st.columns(3)
                dose_panels = [
                    (reference_dose[slice_index], "Reference distribution"),
                    (recommended_dose[slice_index], "AI-recommended plan")
                ]
                for column, (dose_image, title) in zip(panels[:2], dose_panels):
                    with column:
                        figure, axis = plt.subplots(figsize=(4, 4))
                        axis.imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
                        contours = axis.contour(
                            dose_image, levels=[10, 20, 30, 95],
                            colors=["cyan", "yellow", "red", "magenta"],
                            linewidths=1.3
                        )
                        axis.clabel(contours, fmt="%.0f%%", fontsize=7)
                        axis.contour(target_mask, levels=[0.5], colors=["lime"], linewidths=1.2)
                        axis.set_title(title)
                        axis.axis("off")
                        plt.tight_layout()
                        st.pyplot(figure, use_container_width=True)
                        plt.close(figure)
                with panels[2]:
                    figure, axis = plt.subplots(figsize=(4, 4))
                    axis.imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
                    axis.contour(reference_dose[slice_index], levels=[95], colors=["cyan"], linewidths=1.8)
                    axis.contour(recommended_dose[slice_index], levels=[95], colors=["magenta"], linewidths=1.8)
                    axis.contour(target_mask, levels=[0.5], colors=["lime"], linewidths=1.2)
                    axis.set_title("95% isodose comparison")
                    axis.axis("off")
                    plt.tight_layout()
                    st.pyplot(figure, use_container_width=True)
                    plt.close(figure)
                    st.caption("Ideal reference: cyan | AI plan: magenta | Target: green")

                st.markdown("### AI-recommended beam configuration")
                plan_metrics = st.columns(5)
                plan_metrics[0].metric("Energy", f"{result['energy']} MV")
                plan_metrics[1].metric("Beam 1 angle", f"{result['angle_1']:.0f} deg")
                plan_metrics[2].metric("Beam 1 width", f"{result['width_1']} px")
                plan_metrics[3].metric("Beam 2 angle", f"{result['angle_2']:.0f} deg")
                plan_metrics[4].metric("Beam 2 width", f"{result['width_2']} px")


        st.info(
            "The model recommends parameters for the simplified synthetic "
            "beam simulator. The ideal distribution is its reference; results "
            "are educational examples, not clinical plans."
        )


# ============================================================
# RESEARCH CHALLENGE
# ============================================================

elif page == "Research challenge":

    section_header(
        "Research Challenge",
        "Develop a research proposal based on an AI application in radiotherapy."
    )

    st.markdown("### Your research proposal")

    group_name = st.text_input(
        "Research group name"
    )

    application = st.selectbox(
        "Select the application",
        [
            "Image registration",
            "Organ-at-risk segmentation",
            "3D dose prediction"
        ]
    )

    research_problem = st.text_area(
        "1. Define the research problem",
        placeholder="What problem do you want to address?"
    )

    proposed_method = st.text_area(
        "2. Propose an AI methodology",
        placeholder="Which architecture or algorithm would you use?"
    )

    dataset = st.text_area(
        "3. Describe the dataset",
        placeholder="What images, clinical variables or public datasets would you need?"
    )

    evaluation = st.text_area(
        "4. Define evaluation metrics",
        placeholder="How would you evaluate the model?"
    )

    limitations = st.text_area(
        "5. Identify potential limitations",
        placeholder="What are the scientific and clinical limitations?"
    )

    if st.button(
        "Generate research proposal summary",
        type="primary"
    ):

        if (
            research_problem.strip()
            and proposed_method.strip()
        ):

            st.markdown("---")
            st.markdown("## Research proposal summary")

            st.write(f"**Group:** {group_name}")
            st.write(f"**Application:** {application}")

            st.write("**Research problem:**")
            st.write(research_problem)

            st.write("**Proposed methodology:**")
            st.write(proposed_method)

            st.write("**Dataset:**")
            st.write(dataset)

            st.write("**Evaluation:**")
            st.write(evaluation)

            st.write("**Limitations:**")
            st.write(limitations)

        else:

            st.warning(
                "Please complete at least the research problem "
                "and proposed methodology."
            )


# ============================================================
# KNOWLEDGE CHECK
# ============================================================

elif page == "Knowledge check":

    section_header(
        "Knowledge Check",
        "Test your understanding of the concepts introduced in this workshop."
    )

    questions = [

        {
            "question": "What is the main objective of image registration?",
            "options": [
                "To align two or more images",
                "To calculate radiation dose",
                "To remove all image noise",
                "To segment all organs"
            ],
            "answer": "To align two or more images"
        },

        {
            "question": "Which parameters define a 2D rigid transformation?",
            "options": [
                "Translation X, translation Y and rotation",
                "Translation X, translation Y and scaling",
                "Rotation, scaling and deformation",
                "Intensity, contrast and resolution"
            ],
            "answer": "Translation X, translation Y and rotation"
        },

        {
            "question": "What does the Dice coefficient evaluate?",
            "options": [
                "Image acquisition time",
                "Overlap between two segmentation masks",
                "Radiation dose",
                "Image resolution"
            ],
            "answer": "Overlap between two segmentation masks"
        },

        {
            "question": "Does conventional automatic registration necessarily use AI?",
            "options": [
                "Yes, always",
                "No, it can use optimization algorithms",
                "Only when using CT images",
                "Only when using MRI images"
            ],
            "answer": "No, it can use optimization algorithms"
        },

        {
            "question": "What is the main limitation of rigid registration for deforming anatomy?",
            "options": [
                "It cannot model local shape changes",
                "It cannot translate an image",
                "It cannot rotate an image",
                "It always changes image intensity"
            ],
            "answer": "It cannot model local shape changes"
        },

        {
            "question": "What is the purpose of a dose prediction model?",
            "options": [
                "To predict a 3D dose distribution",
                "To acquire CT images",
                "To detect radiation contamination",
                "To measure patient weight"
            ],
            "answer": "To predict a 3D dose distribution"
        }

    ]

    with st.form("knowledge_check_form"):

        answers = []

        for i, question in enumerate(questions):

            st.markdown(f"### Question {i + 1}")

            st.write(question["question"])

            answer = st.radio(
                "Select your answer:",
                question["options"],
                index=None,
                key=f"question_{i}"
            )

            answers.append(answer)

        submitted = st.form_submit_button(
            "Submit answers",
            type="primary"
        )

    if submitted:

        score = sum(
            answer == questions[i]["answer"]
            for i, answer in enumerate(answers)
        )

        st.markdown("---")
        st.markdown("## Your result")

        st.metric(
            "Score",
            f"{score} / {len(questions)}"
        )

        st.progress(
            score / len(questions)
        )

        for i, answer in enumerate(answers):

            correct_answer = questions[i]["answer"]

            st.markdown(
                f"**Question {i + 1}:** "
                f"{'Correct' if answer == correct_answer else 'Incorrect'}"
            )

            if answer != correct_answer:

                st.write(
                    f"Correct answer: {correct_answer}"
                )

        if score == len(questions):

            st.success("All answers are correct.")

        elif score >= 3:

            st.info(
                "Good result. Review the questions you missed."
            )

        else:

            st.warning(
                "Review the workshop sections and try again."
            )


def go_to_workshop_step(main_page, subpage_key=None, subpage=None):
    st.session_state["main_navigation"] = main_page
    st.session_state["scroll_to_top"] = True
    if main_page == "Home":
        st.session_state["registration_subpage"] = "Registration overview"
        st.session_state["segmentation_subpage"] = "Manual segmentation"
        st.session_state["dose_prediction_subpage"] = (
            "3.1 · Beam planning experience"
        )
    if subpage_key is not None:
        st.session_state[subpage_key] = subpage


workshop_steps = [
    ("Home", "Home", "Home", "Workshop introduction.", None, None),
    (
        "Deep learning basics",
        "Deep learning basics",
        "AI basics",
        "Learn the core ideas behind AI and deep learning.",
        None,
        None,
    ),
    (
        "1 · Image registration",
        "1 · Image registration",
        "Registration overview",
        "Learn how CT and MRI images are aligned.",
        "registration_subpage",
        "Registration overview",
    ),
    (
        "4 · AI-based registration",
        "1 · Image registration",
        "AI-based registration",
        "Explore AI-assisted image alignment.",
        "registration_subpage",
        "AI-based registration",
    ),
    (
        "2 · Organ segmentation",
        "2 · Organ segmentation",
        "Manual segmentation",
        "Learn how organs are outlined in medical images.",
        "segmentation_subpage",
        "Manual segmentation",
    ),
    (
        "2 · AI segmentation",
        "2 · Organ segmentation",
        "AI segmentation",
        "Explore AI-assisted organ contouring.",
        "segmentation_subpage",
        "AI segmentation",
    ),
    (
        "3 · Dose prediction",
        "3 · Dose prediction",
        "Beam planning experience",
        "Try arranging treatment beams and inspect the dose.",
        "dose_prediction_subpage",
        "3.1 · Beam planning experience",
    ),
    (
        "3 · AI dose prediction",
        "3 · Dose prediction",
        "AI-based dose prediction",
        "Explore how AI can predict a 3D dose distribution.",
        "dose_prediction_subpage",
        "3.2 · AI-based dose prediction",
    ),
    (
        "Research challenge",
        "Research challenge",
        "Research challenge",
        "Apply the concepts to a radiotherapy research scenario.",
        None,
        None,
    ),
    (
        "Knowledge check",
        "Knowledge check",
        "Knowledge check",
        "Review the key ideas from the workshop.",
        None,
        None,
    ),
]

current_step_index = next(
    (index for index, step in enumerate(workshop_steps) if step[0] == page),
    None,
)

if current_step_index is not None:
    previous_step = (
        workshop_steps[current_step_index - 1]
        if current_step_index > 0
        else None
    )
    next_step = (
        workshop_steps[current_step_index + 1]
        if current_step_index < len(workshop_steps) - 1
        else workshop_steps[0]
    )

    with st.container(horizontal=True, horizontal_alignment="distribute"):
        if previous_step is None:
            st.button(
                "Previous",
                key="previous_step_button",
                disabled=True,
                icon=":material/arrow_back:",
            )
        else:
            st.button(
                "Previous",
                key="previous_step_button",
                icon=":material/arrow_back:",
                on_click=go_to_workshop_step,
                args=(previous_step[1], previous_step[4], previous_step[5]),
            )

        st.button(
            "Next",
            key="next_step_button",
            type="primary",
            icon=":material/arrow_forward:",
            on_click=go_to_workshop_step,
            args=(next_step[1], next_step[4], next_step[5]),
        )

if st.session_state.pop("scroll_to_top", False):
    st.html(
        """
        <script>
        requestAnimationFrame(() => {
            const main = document.querySelector("main");
            if (main) {
                main.scrollTo(0, 0);
                main.scrollTop = 0;
            }
            window.scrollTo(0, 0);
            document.documentElement.scrollTop = 0;
            document.body.scrollTop = 0;
        });
        </script>
        """,
        unsafe_allow_javascript=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "AI in Radiotherapy Workshop | Master 2 Medical Physics"
)

st.caption(
    "Educational prototype – Synthetic data only"
)
