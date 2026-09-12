"""Configuration propagation, package boundaries and callable workflow checks."""

from dataclasses import replace
from pathlib import Path
import ast
import importlib
import json
import os
import subprocess
import sys
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from mrf_diffusion.sequence.definition import (
    MRFSequence,
    SequenceSettings,
    SimulationOptions,
)
from mrf_diffusion.simulation.api import simulate_mrf_signal
from mrf_diffusion.simulation.tissue import TissueParameters
from mrf_diffusion.simulation.signal import (
    simulate_scalar_signal,
    simulate_tensor_signal,
)
from mrf_diffusion.optimization.solver import optimize_sequence
from mrf_diffusion.optimization.settings import SolverSettings
from mrf_diffusion.experiments.settings import (
    ExperimentSettings,
    parse_settings,
    required_inputs,
)
from mrf_diffusion.diffusion.tensor import make_legacy_tensor_grid
from mrf_diffusion.phantoms.checkerboard import make_checkerboard_tensor_phantom
from mrf_diffusion.reconstruction.matching import match_magnitude_dictionary
from tests.cases import ANGLES, PHASES, TENSOR, PREP

ROOT = Path(__file__).resolve().parents[1]


def test_object_input_boundary_preserves_scalar_and_tensor_signals():
    sequence = MRFSequence(ANGLES[:3], PHASES[:3])
    scalar = TissueParameters(1000.0, 80.0, 0.9, 0.001)
    assert_allclose(
        simulate_mrf_signal(scalar, sequence, tensor=False),
        simulate_scalar_signal(ANGLES[:3], PHASES[:3], 1000.0, 80.0, 0.001, 0.9),
        rtol=0,
        atol=0,
    )
    tensor = replace(scalar, diffusion=TENSOR)
    options = SimulationOptions(direction_count=1, state_count=5)
    assert_allclose(
        simulate_mrf_signal(tensor, sequence, options),
        simulate_tensor_signal(
            ANGLES[:3],
            PHASES[:3],
            1000.0,
            80.0,
            0.9,
            TENSOR,
            PREP,
            PREP,
            direction_count=1,
            state_count=5,
        ),
        rtol=0,
        atol=0,
    )


def test_explicit_timing_changes_signal_and_preserves_differentiation():
    args = (ANGLES[:3], PHASES[:3], 1000.0, 80.0, 0.001, 0.9)
    settings = replace(SequenceSettings(), echo_time_ms=6.0)
    baseline = simulate_scalar_signal(*args)
    changed = simulate_scalar_signal(*args, sequence=settings)
    assert not np.allclose(baseline, changed)
    derivative = jax.jacobian(simulate_scalar_signal, argnums=3)(
        *args, sequence=settings
    )
    plus = list(args)
    minus = list(args)
    plus[3] += 0.01
    minus[3] -= 0.01
    finite = (
        simulate_scalar_signal(*plus, sequence=settings)
        - simulate_scalar_signal(*minus, sequence=settings)
    ) / 0.02
    assert_allclose(derivative, finite, rtol=3e-6, atol=1e-9)


@pytest.mark.parametrize(
    "module", ["optimization", "phase_comparison", "bspline_benchmark", "undersampling"]
)
def test_experiment_import_and_dry_run_are_side_effect_free(module, tmp_path):
    # Separate process also detects accidental optional imports or data loading.
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-m", "mrf_diffusion.experiments." + module, "--dry-run"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["length"] > 0
    assert list(tmp_path.iterdir()) == []


def test_config_paths_are_relative_to_config_and_unknown_fields_rejected(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text(
        'input_directory="inputs"\noutput_directory="results"\n[sequence]\necho_time_ms=5.0\n'
    )
    settings = parse_settings("test", ExperimentSettings(), ["--config", str(config)])
    assert Path(settings.input_directory) == tmp_path / "inputs"
    assert settings.sequence.echo_time_ms == 5.0
    assert settings.sequence.repetition_time_ms == 15.0
    with pytest.raises(FileNotFoundError, match="Missing experiment inputs"):
        required_inputs(settings)
    config.write_text("misspelled_parameter=3\n")
    with pytest.raises(ValueError, match="Unknown"):
        parse_settings("test", ExperimentSettings(), ["--config", str(config)])


def test_scientific_layers_do_not_import_workflows():
    forbidden = {
        "epg": {
            "simulation",
            "information",
            "optimization",
            "experiments",
            "visualization",
            "io",
            "sequence",
        },
        "diffusion": {
            "simulation",
            "information",
            "optimization",
            "experiments",
            "visualization",
            "io",
            "sequence",
        },
        "sequence": {
            "simulation",
            "information",
            "optimization",
            "experiments",
            "visualization",
            "io",
        },
        "simulation": {
            "information",
            "optimization",
            "experiments",
            "visualization",
            "io",
        },
        "information": {"optimization", "experiments", "visualization", "io"},
    }
    for layer, banned in forbidden.items():
        for path in (ROOT / "src/mrf_diffusion" / layer).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith("mrf_diffusion.")
                ):
                    assert node.module.split(".")[1] not in banned, (path, node.module)


def test_solver_uses_local_state_and_minimizes_deterministic_quadratic():
    objective = lambda x: jnp.sum((x - jnp.array([0.2, -0.3])) ** 2)
    for _ in range(2):
        result, history = optimize_sequence(
            objective, jnp.array([1.0, 1.0]), (), SolverSettings(max_iterations=10)
        )
        assert result.success
        assert_allclose(result.x, [0.2, -0.3], atol=1e-10)
        assert history[0]["iteration"] == 1


def test_tensor_grid_scales_are_not_mislabeled_as_physical_metrics():
    base = jnp.diag(jnp.array([0.003, 0.002, 0.001]))
    tensors, pairs = make_legacy_tensor_grid(
        base, jnp.array([0.5, 1.0]), jnp.array([0.0, 1.0])
    )
    assert tensors.shape == (2, 2, 3, 3)
    assert_allclose(tensors[0, 0], jnp.diag(jnp.array([0.0015, 0.0001, 0.00005])))
    assert_allclose(tensors[1, 1], base)
    assert_allclose(pairs[0, 1], [0.5, 1.0])


def test_checkerboard_dimensions_mask_and_legacy_scale_limits():
    base = jnp.eye(3)
    phantom = make_checkerboard_tensor_phantom(base, n_tiles=3, radius=100, tile_size=2)
    assert phantom.shape == (6, 6, 3, 3)
    assert_allclose(phantom[0, 0], 0.5 * np.eye(3))
    assert_allclose(phantom[-1, -1], np.diag([1.5, 0.0, 0.0]))
    masked = make_checkerboard_tensor_phantom(base, n_tiles=3, radius=0, tile_size=2)
    assert np.count_nonzero(np.linalg.norm(masked, axis=(-1, -2))) == 1


def test_dictionary_match_recovers_known_entry_and_signal_scale():
    dictionary = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    keys = [(1000.0, 80.0, 0.5, 0.2), (1500.0, 100.0, 0.7, 0.4)]
    parameters, amplitude, correlation = match_magnitude_dictionary(
        dictionary, np.array([0.0, 3.0]), keys, np.array([2.0, 1.5])
    )
    assert parameters == [keys[1]]
    assert_allclose(amplitude, 2.0)
    assert_allclose(correlation, 1.0)


def test_dictionary_accepts_numpy_and_jax_grids_without_changing_signals():
    from mrf_diffusion.reconstruction.dictionary import build_magnitude_dictionary
    from mrf_diffusion.simulation.signal import tensor_signal_magnitude

    options = SimulationOptions(direction_count=1, state_count=5)
    expected = tensor_signal_magnitude(
        ANGLES[:3],
        PHASES[:3],
        1000.0,
        80.0,
        1.0,
        TENSOR,
        PREP,
        PREP,
        direction_count=1,
        state_count=5,
    )
    for array in [np.array, jnp.array]:
        dictionary = build_magnitude_dictionary(
            ANGLES[:3],
            array([1000.0]),
            array([80.0]),
            array([1.0]),
            array([1.0]),
            TENSOR,
            PHASES[:3],
            PREP,
            PREP,
            options,
            SequenceSettings(),
        )
        assert list(dictionary) == [(1000.0, 80.0, 1.0, 1.0)]
        assert_allclose(next(iter(dictionary.values())), expected, rtol=0, atol=0)
