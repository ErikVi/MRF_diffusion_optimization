"""Supplied/generated spiral coordinates and acquisition rotation schedules.

Public arrays use (ky, kx), normally cycles/pixel. No RF or object phase is used.
Rotation is Euclidean in the supplied coordinate units: for unequal pixel sizes,
rotate physical k-space coordinates BEFORE converting each axis to cycles/pixel.
"""

import numpy as np
from ._validation import (
    coordinates_2d,
    image_shape_2d,
    positive_integer,
    real_finite,
    sigpy_module,
)


def rotate_trajectory(coordinates, angle_rad):
    """Rotate (..., samples, 2) (ky,kx) coordinates, retaining units and shape.

    Positive angle sends +kx toward +ky. This is counterclockwise in an x-right,
    y-up Cartesian drawing (clockwise when positive image rows point downward).
    Rotation angles must be scalar radians. Input arrays are not modified.
    """
    coordinates = coordinates_2d(coordinates)
    angle = real_finite(angle_rad, "angle_rad")
    if angle.ndim != 0:
        raise ValueError("angle_rad must be scalar")
    cosine, sine = np.cos(angle), np.sin(angle)
    ky, kx = coordinates[..., 0], coordinates[..., 1]
    return np.stack((sine * kx + cosine * ky, cosine * kx - sine * ky), axis=-1)


def generate_rotated_spiral_trajectory(
    base_interleaf,
    frame_count,
    *,
    interleaves_per_frame=1,
    schedule="uniform",
    angular_increment_rad=2 * np.pi / 32,
    initial_angle_rad=0.0,
    reference_arm_count=32,
):
    """Return (frames, interleaves, samples, 2) coordinates in input units.

    base_interleaf is (samples,2), ordered (ky,kx).
    uniform: angle(frame,arm) = offset + frame*increment + arm*2*pi/L.
    golden: offset - pi*(3-sqrt(5))*(frame*L+arm), matching David's handedness.
    reference_arms: offset + 2*pi/A*(frame + floor(A/L)*arm).
    The latter reproduces David's discrete arm grouping, including unequal gaps
    for L not dividing A. It deliberately differs from uniform interleaves.
    angular_increment_rad applies only to uniform; reference_arm_count to
    reference_arms. No clipping, normalization or density weighting is performed.
    Flatten only the two arm/sample axes when passing a frame to NufftOperator.
    """
    base = coordinates_2d(base_interleaf, single=True)
    frames = positive_integer(frame_count, "frame_count")
    arms = positive_integer(interleaves_per_frame, "interleaves_per_frame")
    offset = real_finite(initial_angle_rad, "initial_angle_rad")
    increment = real_finite(angular_increment_rad, "angular_increment_rad")
    if offset.ndim or increment.ndim:
        raise ValueError("rotation offsets/increments must be scalar")
    if schedule == "uniform":
        angles = offset + np.arange(frames)[:, None] * increment
        angles = angles + np.arange(arms)[None, :] * (2 * np.pi / arms)
    elif schedule == "golden":
        angles = offset - np.pi * (3 - np.sqrt(5)) * np.arange(frames * arms).reshape(
            frames, arms
        )
    elif schedule == "reference_arms":
        count = positive_integer(reference_arm_count, "reference_arm_count")
        if arms > count:
            raise ValueError("interleaves_per_frame cannot exceed reference_arm_count")
        angles = offset + 2 * np.pi / count * (
            np.arange(frames)[:, None] + (count // arms) * np.arange(arms)[None, :]
        )
    else:
        raise ValueError("schedule must be uniform, golden, or reference_arms")
    return np.stack(
        [np.stack([rotate_trajectory(base, angle) for angle in row]) for row in angles]
    )


def to_sigpy_coordinates(coordinates_cycles_per_pixel, image_shape):
    """Convert (...,samples,2) normalized (ky,kx) to SigPy grid coordinates.

    For image (H,W), multiply by (H,W). SigPy uses coordinate components in
    array-axis order: coord[...,0] belongs to rows, coord[...,1] to columns.
    A normalized Nyquist interval [-.5,.5] becomes [-H/2,H/2],[-W/2,W/2].
    Values outside Nyquist are NOT silently clipped; sampling beyond it aliases.
    """
    shape = image_shape_2d(image_shape)
    return coordinates_2d(coordinates_cycles_per_pixel) * np.asarray(shape)


def cartesian_trajectory(image_shape):
    """Return a complete centered Cartesian grid (H*W,2), cycles/pixel.

    Zero frequency is at index floor(size/2) along each axis, for even and odd
    shapes. This is a validation fixture as well as a fully sampled trajectory.
    """
    height, width = image_shape_2d(image_shape)
    ky, kx = np.meshgrid(
        (np.arange(height) - height // 2) / height,
        (np.arange(width) - width // 2) / width,
        indexing="ij",
    )
    return np.stack((ky.ravel(), kx.ravel()), axis=-1)


def load_spiral_coordinates(
    path, *, key="Coords", layout="components_first", component_order="xy"
):
    """Load numeric NPZ coordinates without pickle; return (samples,2) (ky,kx).

    No units are inferred or changed. The caller must know the archive's units.
    David's Single spiral.npz uses Coords (2,1802), xy, normalized cycles/pixel.
    layout: components_first (2,K), samples_first (K,2).
    component_order: xy or yx. This loader does not download or bundle data.
    """
    if layout not in ("components_first", "samples_first"):
        raise ValueError("layout must be components_first or samples_first")
    if component_order not in ("xy", "yx"):
        raise ValueError("component_order must be xy or yx")
    with np.load(path, allow_pickle=False) as archive:
        values = archive[key]
    if layout == "components_first":
        values = values.T
    values = coordinates_2d(values, single=True)
    if component_order == "xy":
        values = values[:, ::-1]
    return values.copy()


def generate_variable_density_spiral(
    *,
    field_of_view_m,
    matrix_size,
    frequency_sampling_factor,
    acceleration,
    density_exponent,
    max_gradient_t_per_m,
    max_slew_t_per_m_per_s,
):
    """Generate one SigPy spiral arm as (samples,2) (ky,kx), cycles/pixel.

    Calls SigPy's analytic variable-density design for an isotropic square FOV.
    Its radial scale matrix_size/(2*FOV) is interpreted as cycles/m, then
    multiplied by pixel size FOV/matrix_size. No unexplained 256 factor is used.
    Timing/gradient waveforms are not returned; this is not scanner certification.
    Multiple arms/rotations are a separate operation.
    """
    size = positive_integer(matrix_size, "matrix_size")
    parameters = (
        field_of_view_m,
        frequency_sampling_factor,
        acceleration,
        density_exponent,
        max_gradient_t_per_m,
        max_slew_t_per_m_per_s,
    )
    values = real_finite(parameters, "spiral design parameters")
    if values.shape != (6,) or np.any(values <= 0):
        raise ValueError("spiral design parameters must be positive scalars")
    if acceleration >= size / 2:
        raise ValueError(
            "acceleration must be below matrix_size/2 for this spiral design"
        )
    sigpy_module()
    from sigpy.mri import spiral

    physical_xy = spiral(
        field_of_view_m,
        size,
        frequency_sampling_factor,
        acceleration,
        1,
        density_exponent,
        max_gradient_t_per_m,
        max_slew_t_per_m_per_s,
    )
    physical_yx = coordinates_2d(physical_xy, single=True)[:, ::-1]
    return physical_yx * (field_of_view_m / size)
