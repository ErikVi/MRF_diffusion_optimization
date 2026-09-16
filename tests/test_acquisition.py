"""Independent spatial encoding checks, before any MRF integration."""

import json
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import convolve

from mrf_diffusion.encoding import (
    NufftOperator,
    cartesian_trajectory,
    rotate_trajectory,
    generate_rotated_spiral_trajectory,
    generate_variable_density_spiral,
    to_sigpy_coordinates,
    load_spiral_coordinates,
    radial_increment_density_compensation,
    pipe_menon_density_compensation,
    calculate_point_spread_function,
    direct_sampling_psf,
)
from mrf_diffusion.reconstruction.images import reconstruct_weighted_adjoint


def fourier_matrix(shape, coordinates):
    """Independent small dense oracle; never call production coordinate helpers."""
    positions = np.array(
        [
            (row - shape[0] // 2, column - shape[1] // 2)
            for row in range(shape[0])
            for column in range(shape[1])
        ]
    )
    return np.exp(-2j * np.pi * (coordinates @ positions.T)) / np.sqrt(np.prod(shape))


def complex_object(shape):
    row, col = np.indices(shape)
    amplitude = ((row >= 1) & (row < shape[0] - 1) & (col >= 2)).astype(float)
    return amplitude * np.exp(1j * (0.17 * row - 0.31 * col))


def test_zero_rotation_preserves_coordinates_without_mutation():
    coords = np.array([[0.1, 0.2], [-0.3, 0.4]])
    result = rotate_trajectory(coords, 0)
    assert_array_equal(result, coords)
    result[0, 0] = 20
    assert coords[0, 0] == 0.1


def test_positive_ninety_degree_rotation():
    # (ky,kx): +x -> +y, +y -> -x.
    coords = np.array([[0, 0.3], [0.2, 0], [-0.2, 0.1]])
    assert_allclose(
        rotate_trajectory(coords, np.pi / 2),
        [[0.3, 0], [0, -0.2], [0.1, 0.2]],
        atol=1e-16,
    )


def test_golden_progression_and_offset():
    trajectory = generate_rotated_spiral_trajectory(
        [[0, 0.25]],
        4,
        interleaves_per_frame=2,
        schedule="golden",
        initial_angle_rad=0.2,
    )
    expected_angles = 0.2 - np.pi * (3 - np.sqrt(5)) * np.arange(8)
    assert trajectory.shape == (4, 2, 1, 2)
    assert_allclose(
        trajectory.reshape(8, 2),
        0.25 * np.column_stack([np.sin(expected_angles), np.cos(expected_angles)]),
    )
    assert_allclose(np.linalg.norm(trajectory, axis=-1), 0.25)


def test_uniform_interleaves_and_frame_increment():
    trajectory = generate_rotated_spiral_trajectory(
        [[0, 0.25]],
        2,
        interleaves_per_frame=4,
        angular_increment_rad=np.pi / 4,
    )
    assert_allclose(
        trajectory[0, :, 0], [[0, 0.25], [0.25, 0], [0, -0.25], [-0.25, 0]], atol=1e-16
    )
    assert_allclose(trajectory[1, 0, 0], np.array([1, 1]) * 0.25 / np.sqrt(2))


def test_reference_arm_schedule_preserves_nondivisor_grouping():
    trajectory = generate_rotated_spiral_trajectory(
        [[0, 0.25]],
        2,
        interleaves_per_frame=3,
        schedule="reference_arms",
    )
    angles = 2 * np.pi / 32 * np.array([[0, 10, 20], [1, 11, 21]])
    assert_allclose(trajectory[..., 0, 0], 0.25 * np.sin(angles))
    assert_allclose(trajectory[..., 0, 1], 0.25 * np.cos(angles))


def test_density_compensation_is_deterministic_per_arm_and_scaled():
    arm = np.array([[0, 0], [0, 0.1], [0, 0.3], [0, 0.2]])
    arms = np.stack([arm, rotate_trajectory(arm, 0.7)])
    weights = radial_increment_density_compensation(arms)
    assert_allclose(weights, [[0, 0.01, 0.06, 0.02]] * 2, atol=1e-16)
    assert_array_equal(weights, radial_increment_density_compensation(arms))
    assert_allclose(radial_increment_density_compensation(arms * 3), weights * 9)
    assert_array_equal(weights[:, 0], 0)


def test_coordinate_conversion_rectangular():
    assert_allclose(to_sigpy_coordinates([[0.25, -0.5]], (8, 10)), [[2, -5]])


def test_encoding_import_is_lazy_and_independent_of_signal_physics():
    import ast
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import mrf_diffusion.encoding; import sys; "
            "assert 'sigpy' not in sys.modules; "
            "assert 'mrf_diffusion.simulation.signal' not in sys.modules",
        ],
        check=True,
    )
    root = Path(__file__).parents[1] / "src" / "mrf_diffusion"
    for area in (
        "epg",
        "diffusion",
        "simulation",
        "sequence",
        "information",
        "optimization",
    ):
        for path in (root / area).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert "encoding" not in (node.module or "").split(".")
                    assert "sigpy" not in (node.module or "").split(".")
                elif isinstance(node, ast.Import):
                    assert all(
                        "sigpy" not in alias.name.split(".")
                        and "encoding" not in alias.name.split(".")
                        for alias in node.names
                    )


def test_default_nufft_accuracy_is_separate_from_high_accuracy_settings():
    shape = (7, 8)
    coords = np.random.default_rng(19).uniform(-0.49, 0.49, (47, 2))
    image = complex_object(shape)
    expected = fourier_matrix(shape, coords) @ image.ravel()
    ordinary = NufftOperator(shape, coords).forward(image)
    accurate = NufftOperator(shape, coords, oversampling=2, kernel_width=6).forward(
        image
    )
    assert np.linalg.norm(ordinary - expected) / np.linalg.norm(expected) < 0.01
    assert np.linalg.norm(accurate - expected) < np.linalg.norm(ordinary - expected)


def test_supplied_archive_axis_order(tmp_path):
    path = tmp_path / "arm.npz"
    np.savez(path, Coords=np.array([[0, 0.1, 0.2], [0.3, 0.2, 0.1]]))
    assert_allclose(load_spiral_coordinates(path), [[0.3, 0], [0.2, 0.1], [0.1, 0.2]])
    with pytest.raises(ValueError):
        load_spiral_coordinates(path, component_order="guess")


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_forward_adjoint_shape_dtype_and_complex_linearity(dtype):
    shape = (5, 8)
    coords = np.random.default_rng(22).uniform(-0.45, 0.45, (31, 2))
    op = NufftOperator(shape, coords, oversampling=2, kernel_width=6)
    image = complex_object(shape).astype(dtype)
    encoded = op.forward(image)
    reconstructed = op.adjoint(encoded)
    assert encoded.shape == (31,)
    assert reconstructed.shape == shape
    assert encoded.dtype == dtype
    assert reconstructed.dtype == dtype
    factor = dtype(0.7 + 1.3j)
    tolerance = 4e-6 if dtype == np.complex64 else 1e-12
    assert_allclose(
        op.forward(factor * image), factor * encoded, atol=tolerance, rtol=tolerance
    )
    assert_allclose(
        op.adjoint(factor * encoded),
        factor * reconstructed,
        atol=tolerance,
        rtol=tolerance,
    )
    assert np.max(np.abs(reconstructed.imag)) > 0.01


@pytest.mark.parametrize("shape", [(5, 8), (6, 7)])
def test_nufft_against_independent_dft_and_adjoint(shape):
    rng = np.random.default_rng(103)
    coords = rng.uniform(-0.49, 0.49, (29, 2))
    op = NufftOperator(shape, coords, oversampling=2, kernel_width=6)
    image = complex_object(shape)
    samples = rng.normal(size=29) + 1j * rng.normal(size=29)
    matrix = fourier_matrix(shape, coords)
    assert_allclose(op.forward(image), matrix @ image.ravel(), atol=2e-5, rtol=2e-5)
    assert_allclose(
        op.adjoint(samples).ravel(), matrix.conj().T @ samples, atol=2e-5, rtol=2e-5
    )
    assert_allclose(
        np.vdot(op.forward(image), samples),
        np.vdot(image, op.adjoint(samples)),
        atol=2e-12,
        rtol=2e-12,
    )


@pytest.mark.parametrize("shape", [(7, 9), (6, 8), (5, 8)])
def test_fully_sampled_complex_object_reconstruction(shape):
    op = NufftOperator(
        shape, cartesian_trajectory(shape), oversampling=2, kernel_width=6
    )
    image = complex_object(shape)
    result = reconstruct_weighted_adjoint(op, op.forward(image))
    assert_allclose(result, image, atol=3e-5, rtol=3e-5)
    # Absolute gain is part of the test, not fitted away.
    assert_allclose(np.linalg.norm(result), np.linalg.norm(image), rtol=2e-5)


def test_off_center_impulse_has_expected_phase_ramp():
    shape = (5, 8)
    coordinates = np.array([[0, 0.1], [0.2, 0], [0.11, -0.23]])
    image = np.zeros(shape, complex)
    image[1, 6] = 1j
    expected = (
        1j
        * np.exp(-2j * np.pi * (-coordinates[:, 0] + 2 * coordinates[:, 1]))
        / np.sqrt(40)
    )
    op = NufftOperator(shape, coordinates, oversampling=2, kernel_width=6)
    assert_allclose(op.forward(image), expected, atol=3e-6)


def test_identical_trajectory_reproducibility_zero_and_input_ownership():
    coordinates = cartesian_trajectory((5, 6))
    op = NufftOperator((5, 6), coordinates)
    other = NufftOperator((5, 6), coordinates.copy())
    coordinates[:] = 0
    image = complex_object((5, 6))
    saved = image.copy()
    assert_array_equal(op.forward(image), other.forward(image))
    assert_array_equal(image, saved)
    assert_array_equal(op.forward(np.zeros((5, 6), complex)), np.zeros(30))
    assert_array_equal(op.adjoint(np.zeros(30, complex)), np.zeros((5, 6)))


def test_weighting_applied_once_and_without_gain_normalization():
    op = NufftOperator((5, 6), cartesian_trajectory((5, 6)))
    samples = op.forward(complex_object((5, 6)))
    weights = np.linspace(0.2, 1.2, len(samples))
    result = reconstruct_weighted_adjoint(op, samples, weights)
    assert_allclose(result, op.adjoint(samples * weights), atol=0)
    assert_allclose(
        reconstruct_weighted_adjoint(op, samples, 3 * weights), 3 * result, atol=1e-14
    )
    assert_array_equal(
        reconstruct_weighted_adjoint(op, samples, np.zeros(len(samples))), 0
    )


def test_psf_matches_dense_normal_operator_with_absolute_scale():
    shape = (5, 6)
    coords = np.random.default_rng(4).uniform(-0.45, 0.45, (41, 2))
    weights = np.linspace(0.2, 1.3, len(coords))
    matrix = fourier_matrix(shape, coords)
    impulse = np.zeros(shape)
    impulse[2, 3] = 1
    expected = (matrix.conj().T @ (weights * (matrix @ impulse.ravel()))).reshape(shape)
    assert_allclose(direct_sampling_psf(coords, shape, weights), expected, atol=2e-15)
    op = NufftOperator(shape, coords, oversampling=2, kernel_width=6)
    assert_allclose(calculate_point_spread_function(op, weights), expected, atol=2e-5)
    assert_allclose(expected[2, 3], weights.sum() / 30)


def test_extended_psf_linear_convolution_matches_direct_encoding():
    shape = (4, 5)
    coords = np.random.default_rng(5).uniform(-0.45, 0.45, (27, 2))
    weights = np.linspace(0.1, 1, 27)
    image = complex_object(shape)
    matrix = fourier_matrix(shape, coords)
    expected = (matrix.conj().T @ (weights * (matrix @ image.ravel()))).reshape(shape)
    kernel = direct_sampling_psf(coords, shape, weights, displacement_shape=(7, 9))
    assert_allclose(convolve(image, kernel, mode="same"), expected, atol=3e-15)


def test_pipe_menon_is_explicit_separate_finite_method():
    weights = pipe_menon_density_compensation(
        cartesian_trajectory((5, 5)), (5, 5), iterations=3
    )
    assert weights.shape == (25,)
    assert np.all(np.isfinite(weights)) and np.all(weights > 0)
    assert_array_equal(
        weights,
        pipe_menon_density_compensation(
            cartesian_trajectory((5, 5)), (5, 5), iterations=3
        ),
    )


def test_variable_density_design_scale_and_determinism():
    kwargs = dict(
        field_of_view_m=0.24,
        matrix_size=120,
        frequency_sampling_factor=0.25,
        acceleration=32,
        density_exponent=6,
        max_gradient_t_per_m=0.03,
        max_slew_t_per_m_per_s=150,
    )
    coords = generate_variable_density_spiral(**kwargs)
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert np.all(np.isfinite(coords))
    assert_allclose(np.linalg.norm(coords[-1]), 0.5, atol=1e-12)
    assert_array_equal(coords, generate_variable_density_spiral(**kwargs))


@pytest.mark.parametrize("coords", [[], [[0, 1, 2]], [[np.nan, 0]], [[1j, 0]]])
def test_invalid_coordinates_rejected(coords):
    with pytest.raises(ValueError):
        NufftOperator((5, 5), coords)


def test_invalid_shapes_weights_and_schedule_rejected():
    op = NufftOperator((5, 5), [[0, 0]])
    for shape in [(0, 5), (5.5, 6), (5, 6, 7)]:
        with pytest.raises(ValueError):
            NufftOperator(shape, [[0, 0]])
    with pytest.raises(ValueError):
        op.forward(np.zeros((5, 5, 1)))
    with pytest.raises(ValueError):
        op.adjoint(np.zeros((1, 1)))
    with pytest.raises(ValueError):
        reconstruct_weighted_adjoint(op, [1], [-1])
    with pytest.raises(ValueError):
        reconstruct_weighted_adjoint(op, [1], [[1]])
    with pytest.raises(ValueError):
        generate_rotated_spiral_trajectory([[0, 0]], 1, schedule="unknown")
    with pytest.raises(ValueError):
        generate_rotated_spiral_trajectory(
            [[0, 0]], 1, schedule="reference_arms", interleaves_per_frame=33
        )


def test_reference_method_frozen_outputs():
    """Values produced by actual external helpers, not re-created by this test."""
    path = Path(__file__).parent / "reference" / "acquisition_external.json"
    reference = json.loads(path.read_text())
    arm = np.array(reference["base_arm_yx_cycles_per_pixel"])
    for schedule in ("golden", "reference_arms"):
        actual = generate_rotated_spiral_trajectory(
            arm,
            2,
            interleaves_per_frame=3,
            schedule=schedule,
            initial_angle_rad=0.17,
        )
        assert_allclose(actual, reference[schedule], atol=2e-16)
    assert_allclose(
        radial_increment_density_compensation(arm),
        reference["radial_weights"],
        atol=2e-17,
    )
    expected = np.array(reference["direct_psf_real"]) + 1j * np.array(
        reference["direct_psf_imag"]
    )
    assert_allclose(
        direct_sampling_psf(arm, (5, 5), reference["radial_weights"]),
        expected,
        atol=2e-16,
    )
