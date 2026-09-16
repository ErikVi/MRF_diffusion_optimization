"""Physical tensor maps and forward-only complex MRF integration."""

from dataclasses import replace
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from mrf_diffusion.diffusion.parameterization import (
    axisymmetric_tensor,
    tensor_invariants,
)
from mrf_diffusion.phantoms.geometry import (
    checkerboard_labels,
    circular_support,
    quadratic_object_phase_map,
)
from mrf_diffusion.phantoms.maps import assign_tissue_maps
from mrf_diffusion.simulation.image_series import simulate_phantom_image_series
from mrf_diffusion.simulation.api import simulate_mrf_signal
from mrf_diffusion.simulation.tissue import TissueParameters
from mrf_diffusion.sequence.definition import (
    MRFSequence,
    SequenceSettings,
    SimulationOptions,
)
from mrf_diffusion.encoding.series import acquire_image_series
from mrf_diffusion.encoding.trajectory import (
    cartesian_trajectory,
    generate_rotated_spiral_trajectory,
)
from mrf_diffusion.reconstruction.series import reconstruct_acquisition_frames
from mrf_diffusion.experiments.forward_settings import ForwardSettings
from mrf_diffusion.experiments.forward_undersampling import build_sequence, run

OPTIONS = SimulationOptions(direction_count=2, state_count=20)
SEQUENCE = MRFSequence(
    np.array([0.2, 0.3, 0.4, 0.25]), np.array([0.0, 0.2, -0.4, 0.7]), SequenceSettings()
)


def phantom(**overrides):
    kwargs = dict(
        t1_ms=(1000.0, 1000.0),
        t2_ms=(80.0, 80.0),
        proton_density=(1.0, 1.0),
        mean_diffusivity=(0.0008, 0.0008),
        fractional_anisotropy=(0.3, 0.3),
        principal_directions=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    kwargs.update(overrides)
    return assign_tissue_maps(
        checkerboard_labels((4, 4), (2, 2)), circular_support((4, 4)), **kwargs
    )


@pytest.mark.parametrize("md", [0.0002, 0.001, 0.003])
@pytest.mark.parametrize("fa", [0.0, 0.2, 0.7, 1.0])
def test_tensor_reproduces_requested_md_fa(md, fa):
    tensor = axisymmetric_tensor(md, fa, [1, 2, 3])
    values = np.linalg.eigvalsh(tensor)
    assert values.min() > -1e-17
    assert_allclose(values.mean(), md, atol=1e-17)
    independent = np.sqrt(1.5 * np.sum((values - md) ** 2) / np.sum(values**2))
    assert_allclose(independent, fa, atol=1e-14)
    measured_md, measured_fa, _ = tensor_invariants(tensor)
    assert_allclose([measured_md, measured_fa], [md, fa], atol=1e-14)
    assert_allclose(tensor, axisymmetric_tensor(md, fa, [-1, -2, -3]))


def test_invalid_tensors_rejected():
    for md, fa, direction in [
        (-1, 0.2, [1, 0, 0]),
        (0.001, 1.1, [1, 0, 0]),
        (0.001, 0.2, [0, 0, 0]),
    ]:
        with pytest.raises(ValueError):
            axisymmetric_tensor(md, fa, direction)
    with pytest.raises(ValueError):
        tensor_invariants(np.diag([0.001, 0.002, -0.001]))


def test_homogeneous_unique_cache_matches_exact_simulator(monkeypatch):
    import mrf_diffusion.simulation.image_series as module

    actual = module.simulate_mrf_signal
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return actual(*args, **kwargs)

    monkeypatch.setattr(module, "simulate_mrf_signal", counted)
    maps = phantom()
    result = simulate_phantom_image_series(maps, SEQUENCE, OPTIONS)
    raw = np.asarray(
        actual(
            TissueParameters(1000.0, 80.0, 1.0, maps.diffusion_tensor[0, 0]),
            SEQUENCE,
            OPTIONS,
        )
    )
    assert len(calls) == result.unique_signal_count == 1
    assert result.images.shape == (8, 4, 4)
    assert_allclose(result.images[:, 0, 0], raw[0] + 1j * raw[1], atol=0)
    assert_allclose(result.images[:, 3, 3], result.images[:, 0, 0])
    assert_array_equal(result.direction_index, [0] * 4 + [1] * 4)
    assert_array_equal(result.readout_index, list(range(4)) * 2)


@pytest.mark.parametrize(
    "changed",
    [
        {"t1_ms": (700.0, 1400.0)},
        {"t2_ms": (40.0, 120.0)},
        {"mean_diffusivity": (0.0004, 0.0012)},
        {"fractional_anisotropy": (0.1, 0.8)},
        {
            "t1_ms": (700.0, 1400.0),
            "t2_ms": (50.0, 100.0),
            "mean_diffusivity": (0.0005, 0.001),
            "fractional_anisotropy": (0.2, 0.7),
        },
    ],
)
def test_contrast_enters_actual_voxel_simulation(changed):
    maps = phantom(**changed)
    result = simulate_phantom_image_series(maps, SEQUENCE, OPTIONS)
    assert result.unique_signal_count == 2
    assert not np.allclose(result.images[:, 0, 0], result.images[:, 0, 2])
    for row, column in [(0, 0), (0, 2)]:
        raw = np.asarray(
            simulate_mrf_signal(
                TissueParameters(
                    float(maps.t1_ms[row, column]),
                    float(maps.t2_ms[row, column]),
                    1.0,
                    maps.diffusion_tensor[row, column],
                ),
                SEQUENCE,
                OPTIONS,
            )
        )
        assert_allclose(result.images[:, row, column], raw[0] + 1j * raw[1], atol=0)


def test_md_diffusion_sensitive_first_echo_limit():
    maps = phantom(
        t1_ms=(1e12, 1e12),
        t2_ms=(1e12, 1e12),
        mean_diffusivity=(0.0002, 0.0012),
        fractional_anisotropy=(0.0, 0.0),
    )
    result = simulate_phantom_image_series(maps, SEQUENCE, OPTIONS)
    assert abs(result.images[0, 0, 2]) < abs(result.images[0, 0, 0])


def test_density_and_object_phase_applied_once():
    original = simulate_phantom_image_series(phantom(), SEQUENCE, OPTIONS)
    object_phase_map = quadratic_object_phase_map((4, 4), 1.3)
    altered = simulate_phantom_image_series(
        phantom(proton_density=(0.5, 2.0), object_phase_map=object_phase_map),
        SEQUENCE,
        OPTIONS,
    )
    scale = np.where(checkerboard_labels((4, 4), (2, 2)) == 0, 0.5, 2.0) * np.exp(
        1j * object_phase_map
    )
    assert altered.unique_signal_count == 1
    assert_allclose(altered.images, original.images * scale[None], atol=1e-15)


@pytest.mark.parametrize("mode", ["none", "reference", "explicit", "optimized"])
def test_actual_rf_train_used_once(mode, tmp_path):
    settings = ForwardSettings(input_directory=str(tmp_path))
    train = replace(
        settings.sequence,
        flip_angles_rad=tuple(SEQUENCE.flip_angles),
        rf_phase_mode=mode,
        rf_phases_rad=tuple(SEQUENCE.rf_phases),
    )
    if mode == "optimized":
        np.savez(
            tmp_path / "paired.npz",
            flip_angles_rad=SEQUENCE.flip_angles,
            rf_phases_rad=SEQUENCE.rf_phases,
        )
        train = replace(train, optimized_archive="paired.npz")
    sequence = build_sequence(replace(settings, sequence=train))
    result = simulate_phantom_image_series(phantom(), sequence, OPTIONS)
    raw = np.asarray(
        simulate_mrf_signal(
            TissueParameters(1000.0, 80.0, 1.0, phantom().diffusion_tensor[0, 0]),
            sequence,
            OPTIONS,
        )
    )
    assert_allclose(result.images[:, 0, 0], raw[0] + 1j * raw[1], atol=0)
    if mode == "none":
        assert_array_equal(sequence.rf_phases, 0)
    if mode in ("explicit", "optimized"):
        assert_array_equal(sequence.rf_phases, SEQUENCE.rf_phases)
        zero = simulate_phantom_image_series(
            phantom(), replace(sequence, rf_phases=np.zeros(4)), OPTIONS
        )
        assert not np.allclose(result.images, zero.images)


def test_missing_optimized_sequence_not_fabricated():
    settings = ForwardSettings()
    with pytest.raises(ValueError, match="paired"):
        build_sequence(
            replace(
                settings, sequence=replace(settings.sequence, rf_phase_mode="optimized")
            )
        )


@pytest.mark.parametrize("separate", [False, True])
def test_inversion_slicing_metadata_matches_signal(separate):
    sequence = replace(
        SEQUENCE,
        settings=replace(
            SEQUENCE.settings, reuse_readout_train_for_inversion=not separate
        ),
        preparation_flip_angles=np.array([0.2, 0.4]),
        preparation_phases=np.array([0.1, -0.3]),
    )
    options = replace(
        OPTIONS,
        include_inversion=True,
        sampling=True,
        sampling_offset=1,
        sampling_rate=2,
    )
    result = simulate_phantom_image_series(phantom(), sequence, options)
    prefix = 3 if separate else 5
    expected = np.arange(prefix + 4)[1::2]
    assert_array_equal(result.source_sample_index, expected)
    assert_array_equal(result.direction_index, np.where(expected < prefix, 0, 1))
    assert result.images.shape[0] == len(expected)


def test_cartesian_complex_frame_recovery():
    images = simulate_phantom_image_series(
        phantom(object_phase_map=np.full((4, 4), 0.6)), SEQUENCE, OPTIONS
    ).images
    grid = cartesian_trajectory((4, 4))
    coords = np.broadcast_to(grid, (len(images), 1, *grid.shape))
    acquisition = acquire_image_series(images, coords, oversampling=2, kernel_width=6)
    recovered = reconstruct_acquisition_frames(acquisition)
    assert_allclose(recovered, images, atol=2e-6, rtol=3e-5)


def test_spiral_interleaves_complex_scaling_and_seeded_noise():
    images = simulate_phantom_image_series(phantom(), SEQUENCE, OPTIONS).images
    arm = np.array([[0, 0], [0.04, 0.1], [0.2, 0.1], [0.3, -0.2]])
    coords = generate_rotated_spiral_trajectory(
        arm, len(images), interleaves_per_frame=2, schedule="golden"
    )
    clean = acquire_image_series(images, coords)
    assert clean.kspace.shape == (8, 2, 4)
    changed = acquire_image_series(1j * images, coords)
    assert_allclose(changed.kspace, 1j * clean.kspace, atol=1e-15)
    noisy = acquire_image_series(
        images, coords, noise_std_per_channel=0.001, noise_seed=14
    )
    again = acquire_image_series(
        images, coords, noise_std_per_channel=0.001, noise_seed=14
    )
    assert_array_equal(noisy.kspace, again.kspace)
    assert not np.allclose(
        noisy.kspace - clean.kspace, (noisy.kspace - clean.kspace)[0]
    )
    assert (
        reconstruct_acquisition_frames(
            clean, density_compensation="radial_increment"
        ).shape
        == images.shape
    )


def test_short_fixture_state_convergence():
    maps = phantom()
    a = simulate_phantom_image_series(maps, SEQUENCE, OPTIONS).images
    b = simulate_phantom_image_series(
        maps, SEQUENCE, replace(OPTIONS, state_count=40)
    ).images
    assert_allclose(a, b, atol=1e-12, rtol=1e-12)


def test_runner_outputs_without_matching(tmp_path):
    settings = ForwardSettings(output_directory=str(tmp_path / "out"))
    settings = replace(
        settings,
        phantom=replace(settings.phantom, image_shape=(4, 4), tile_shape=(2, 2)),
        trajectory=replace(settings.trajectory, kind="cartesian"),
        sequence=replace(
            settings.sequence,
            flip_angles_rad=tuple(SEQUENCE.flip_angles),
            rf_phase_mode="none",
        ),
        reconstruction=replace(settings.reconstruction, representative_frames=(0,)),
    )
    maps, series, acquisition, reconstruction = run(settings)
    assert_allclose(
        np.load(tmp_path / "out/ground_truth_md_mm2_per_s.npy"),
        maps.mean_diffusivity_map,
    )
    assert (tmp_path / "out/signal_phase_0.png").is_file()
    assert (tmp_path / "out/adjoint_magnitude_0.png").is_file()
    assert not list((tmp_path / "out").glob("*matched*"))
    assert (
        json.loads((tmp_path / "out/run.json").read_text())["result"][
            "unique_signal_count"
        ]
        == 2
    )
