"""Recovery tests use analytical limits and the authoritative tensor simulator."""

from dataclasses import replace
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from mrf_diffusion.experiments.quantitative_settings import QuantitativeSettings
from mrf_diffusion.experiments.forward_undersampling import (
    build_phantom,
    build_sequence,
    build_trajectories,
)
from mrf_diffusion.experiments.quantitative_undersampling import (
    reference_recovery_gate,
    load_comparison_sequences,
    run,
)
from mrf_diffusion.reconstruction.tensor_dictionary import (
    DictionaryGrid,
    TensorDictionary,
    estimate_dictionary_size,
    generate_tensor_dictionary,
)
from mrf_diffusion.reconstruction.quantitative import match_complex_dictionary
from mrf_diffusion.reconstruction.metrics import (
    parameter_errors,
    evaluate_parameter_maps,
)
from mrf_diffusion.reconstruction.calibrated import reconstruct_with_impulse_gain
from mrf_diffusion.simulation.image_series import simulate_phantom_image_series
from mrf_diffusion.encoding.series import acquire_image_series
from mrf_diffusion.encoding.trajectory import cartesian_trajectory


@pytest.fixture(scope="module")
def fixture():
    settings = QuantitativeSettings()
    settings = replace(
        settings,
        phantom=replace(settings.phantom, image_shape=(4, 4), tile_shape=(2, 2)),
    )
    sequence = build_sequence(settings)
    dictionary = generate_tensor_dictionary(
        settings.dictionary, sequence, settings.simulation
    )
    return settings, sequence, dictionary


def test_dictionary_preflight_and_no_density_axis(monkeypatch, fixture):
    settings, sequence, dictionary = fixture
    estimate = estimate_dictionary_size(settings.dictionary, 16)
    assert estimate["entries_upper_bound"] == 32
    assert estimate["signal_bytes"] == 32 * 16 * 16
    assert estimate["amplitude_samples"] == 0
    import mrf_diffusion.reconstruction.tensor_dictionary as module

    monkeypatch.setattr(
        module,
        "simulate_mrf_signal",
        lambda *a, **k: pytest.fail("must reject before simulation"),
    )
    with pytest.raises(ValueError, match="budget"):
        generate_tensor_dictionary(
            replace(settings.dictionary, max_entries=2), sequence, settings.simulation
        )
    with pytest.raises(ValueError, match="budget"):
        generate_tensor_dictionary(
            replace(settings.dictionary, max_working_bytes=10),
            sequence,
            settings.simulation,
        )


def test_complex_self_match_all_atoms_density_and_phase(fixture):
    _, _, dictionary = fixture
    beta = np.linspace(0.4, 1.4, len(dictionary.signals)) * np.exp(
        1j * np.linspace(-2, 2, len(dictionary.signals))
    )
    images = (dictionary.signals * beta[:, None]).T[:, None, :]
    result = match_complex_dictionary(
        images, dictionary, np.ones(images.shape[1:], bool), batch_size=3
    )
    assert_array_equal(result.dictionary_index.ravel(), np.arange(len(beta)))
    assert_allclose(result.complex_scale.ravel(), beta, atol=1e-12)
    assert not result.ambiguous.any()
    assert_allclose(result.relative_residual, 0, atol=1e-12)
    assert_allclose(
        result.maps["md_mm2_per_s"].ravel(),
        np.trace(dictionary.tensors, axis1=1, axis2=2) / 3,
    )
    eigenvalues = np.linalg.eigvalsh(dictionary.tensors)
    md = eigenvalues.mean(axis=1)
    expected_fa = np.sqrt(
        1.5
        * np.sum((eigenvalues - md[:, None]) ** 2, axis=1)
        / np.sum(eigenvalues**2, axis=1)
    )
    assert_allclose(result.maps["fa"].ravel(), expected_fa, atol=1e-12)


def test_phase_matching_not_magnitude_matching():
    # Same magnitudes, different temporal phase: only complex matching distinguishes.
    signals = np.array([[1, 1j, -1], [1, -1j, -1]], complex)
    dictionary = TensorDictionary(
        signals,
        np.array([1000.0, 1200.0]),
        np.array([80.0, 90.0]),
        np.stack([np.eye(3) * 0.001] * 2),
    )
    result = match_complex_dictionary(
        (signals[1] * 2 * np.exp(0.7j))[:, None, None],
        dictionary,
        np.ones((1, 1), bool),
    )
    assert result.dictionary_index.item() == 1
    assert_allclose(result.complex_scale, 2 * np.exp(0.7j))


def test_zero_nonfinite_background_and_ambiguous():
    dictionary = TensorDictionary(
        np.array([[1, 1j], [2, 2j]]),
        np.array([1000.0, 1500.0]),
        np.array([80.0, 80.0]),
        np.stack([np.eye(3) * 0.001] * 2),
    )
    images = np.array([[[1, 0, np.nan, 1]], [[1j, 0, 0, 1j]]])
    result = match_complex_dictionary(
        images, dictionary, np.array([[True, True, True, False]])
    )
    assert_array_equal(result.valid, [[True, False, False, False]])
    assert result.ambiguous[0, 0]
    assert np.all(result.dictionary_index[0, 1:] == -1)
    assert np.all(np.isnan(result.maps["t1_ms"][0, 1:]))


@pytest.mark.parametrize(
    "contrast", ["homogeneous", "t1", "t2", "md", "fa", "combined"]
)
def test_on_grid_phantom_recovery(fixture, contrast):
    settings, sequence, dictionary = fixture
    p = replace(
        settings.phantom,
        t1_ms=(750.0, 750.0),
        t2_ms=(70.0, 70.0),
        proton_density=(0.7, 0.7),
    )
    d = replace(
        settings.diffusion,
        mean_diffusivity_mm2_per_s=(0.0007, 0.0007),
        fractional_anisotropy=(0.2, 0.2),
        principal_directions_xyz=((1.0, 0.0, 0.0),) * 2,
    )
    if contrast in ("t1", "combined"):
        p = replace(p, t1_ms=(750.0, 1250.0))
    if contrast in ("t2", "combined"):
        p = replace(p, t2_ms=(70.0, 90.0))
    if contrast in ("md", "combined"):
        d = replace(d, mean_diffusivity_mm2_per_s=(0.0007, 0.001))
    if contrast in ("fa", "combined"):
        d = replace(d, fractional_anisotropy=(0.2, 0.7))
    settings = replace(
        settings,
        phantom=p,
        diffusion=d,
        object_phase=replace(settings.object_phase, enabled=True, edge_phase_rad=1.2),
    )
    phantom = build_phantom(settings)
    series = simulate_phantom_image_series(phantom, sequence, settings.simulation)
    coordinates = build_trajectories(
        replace(settings, trajectory=replace(settings.trajectory, kind="cartesian")),
        len(series.images),
    )
    gate = reference_recovery_gate(settings, phantom, series, dictionary, coordinates)
    assert gate["passed"], gate


def test_wrong_rf_train_does_not_pass_signal_agreement(fixture):
    settings, sequence, dictionary = fixture
    wrong = replace(sequence, rf_phases=np.zeros(len(sequence.rf_phases)))
    phantom = build_phantom(settings)
    images = simulate_phantom_image_series(phantom, wrong, settings.simulation).images
    result = match_complex_dictionary(images, dictionary, phantom.support)
    assert np.min(result.relative_residual) > 1e-3


def test_off_grid_gate_rejects_before_undersampling(tmp_path, monkeypatch, fixture):
    settings, _, _ = fixture
    import mrf_diffusion.experiments.quantitative_undersampling as module

    original = module.build_trajectories

    def checked(settings, frames):
        assert settings.trajectory.kind == "cartesian"
        return original(settings, frames)

    monkeypatch.setattr(module, "build_trajectories", checked)
    settings = replace(
        settings,
        output_directory=str(tmp_path),
        phantom=replace(settings.phantom, t1_ms=(900.0, 1100.0)),
    )
    with pytest.raises(ValueError, match="fully sampled recovery failed"):
        run(settings)
    report = json.loads((tmp_path / "status.json").read_text())
    assert report["status"] == "reference_recovery_failed"
    assert not (tmp_path / "metrics.json").exists()


def test_impulse_gain_calibration_is_not_truth_scaling():
    rng = np.random.default_rng(2)
    images = rng.normal(size=(2, 4, 4)) + 1j * rng.normal(size=(2, 4, 4))
    coordinates = np.broadcast_to(cartesian_trajectory((4, 4)), (2, 2, 16, 2))
    acquired = acquire_image_series(images, coordinates, oversampling=2, kernel_width=6)
    raw, gains = reconstruct_with_impulse_gain(acquired)
    corrected, gain = reconstruct_with_impulse_gain(acquired, unit_impulse_gain=True)
    assert_allclose(raw, 2 * images, atol=1e-4)
    assert_allclose(gain, 2, atol=1e-4)
    assert_allclose(corrected, images, atol=5e-5)


def test_metrics_analytic_bias_rmse_and_invalid():
    truth = np.array([1.0, 3.0])
    metrics = parameter_errors(truth, truth + 2, np.ones(2, bool))
    assert metrics["mae"] == metrics["rmse"] == metrics["bias"] == 2
    assert_allclose(metrics["nrmse"], 2 / np.sqrt(5))
    assert parameter_errors(np.zeros(2), np.ones(2), np.ones(2, bool))["nrmse"] is None
    metrics = parameter_errors(truth, np.array([np.nan, 3.0]), np.ones(2, bool))
    assert metrics["valid_voxels"] == metrics["invalid_voxels"] == 1
    regions = evaluate_parameter_maps(
        {"fa": truth}, {"fa": truth + 2}, np.ones(2, bool), np.array([0, 1])
    )
    assert regions["region_1"]["fa"]["bias"] == 2


def test_absent_optimized_sequences_are_not_fabricated(fixture):
    settings, _, _ = fixture
    sequences, missing, _ = load_comparison_sequences(settings)
    assert list(sequences) == ["baseline"]
    assert missing == ["flip_optimized", "joint_optimized"]


def test_paired_archives_and_equal_noise_comparison(tmp_path, monkeypatch, fixture):
    settings, sequence, _ = fixture
    # Identical controlled sequences test the comparison mechanism, not optimization.
    np.savez(
        tmp_path / "paired.npz",
        flip_angles_rad=sequence.flip_angles,
        rf_phases_rad=sequence.rf_phases,
    )
    config = tmp_path / "candidate.toml"
    config.write_text(
        'input_directory="."\n[sequence]\nrf_phase_mode="optimized"\noptimized_archive="paired.npz"\n[simulation]\ndirection_count=2\nstate_count=20\n'
    )
    import mrf_diffusion.visualization.quantitative as plots

    monkeypatch.setattr(plots, "plot_parameter_triptychs", lambda *a: None)
    monkeypatch.setattr(plots, "plot_sampling_comparison", lambda *a: None)
    settings = replace(
        settings,
        input_directory=str(tmp_path),
        output_directory=str(tmp_path / "results"),
        trajectory=replace(settings.trajectory, design_matrix_size=4, acceleration=1.0),
        comparison=replace(
            settings.comparison,
            joint_optimized_config="candidate.toml",
            interleaf_counts=(1,),
            noise_std_per_channel=(0.0, 0.001),
        ),
    )
    records, gates = run(settings)
    assert all(g["passed"] for g in gates.values())
    for baseline in [r for r in records if r["sequence"] == "baseline"]:
        joint = next(
            r
            for r in records
            if r["sequence"] == "joint_optimized"
            and r["condition"] == baseline["condition"]
        )
        assert baseline["metrics"] == joint["metrics"]
        first = np.load(
            tmp_path / "results" / "baseline" / baseline["condition"] / "kspace.npy"
        )
        second = np.load(
            tmp_path / "results" / "joint_optimized" / joint["condition"] / "kspace.npy"
        )
        assert_array_equal(first, second)
    status = json.loads((tmp_path / "results/status.json").read_text())
    assert status["missing_sequences"] == ["flip_optimized"]
    assert (tmp_path / "results/metrics.csv").is_file()
