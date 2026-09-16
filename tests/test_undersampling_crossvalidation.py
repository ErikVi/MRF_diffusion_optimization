"""Frozen external signal regressions; no external software required at test time."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
from mrf_diffusion.sequence.definition import (
    MRFSequence,
    SequenceSettings,
    SimulationOptions,
)
from mrf_diffusion.simulation.api import simulate_mrf_signal
from mrf_diffusion.simulation.tissue import TissueParameters

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads(
    (ROOT / "tests/reference/undersampling_common_signal.json").read_text()
)


def common_signal(*, scalar=False, density=1.0, normal_sequence=False):
    settings = (
        SequenceSettings()
        if normal_sequence
        else replace(
            SequenceSettings(),
            echo_time_ms=0.0,
            preparation_angles=(0.0, 0.0, 0.0),
            initial_longitudinal_magnetization=density,
        )
    )
    sequence = MRFSequence(np.array(REFERENCE["angles_rad"]), np.zeros(12), settings)
    channels = np.asarray(
        simulate_mrf_signal(
            TissueParameters(
                1000.0, 80.0, density, 0.0 if scalar else np.zeros((3, 3))
            ),
            sequence,
            SimulationOptions(direction_count=1, state_count=32),
            tensor=not scalar,
        )
    )
    return channels[0] + 1j * channels[1]


def external_signal():
    return np.array(REFERENCE["external_signal_real"]) + 1j * np.array(
        REFERENCE["external_signal_imag"]
    )


def test_tensor_common_case_matches_pinned_external_signal():
    assert_allclose(common_signal(), external_signal(), rtol=0, atol=1e-12)


def test_scalar_common_case_accounts_for_existing_float32_rf():
    assert_allclose(common_signal(scalar=True), external_signal(), rtol=0, atol=1e-6)


@pytest.mark.parametrize("density", [0.8, 1.3])
def test_common_m0_requires_matching_initial_magnetization(density):
    assert_allclose(
        common_signal(density=density), density * external_signal(), rtol=0, atol=1e-12
    )


def test_normal_prepared_sequence_is_not_silently_treated_as_external_sequence():
    assert (
        np.linalg.norm(common_signal(normal_sequence=True) - external_signal())
        / np.linalg.norm(external_signal())
        > 1
    )


def test_reference_loader_rejects_changed_source_before_execution(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "crossvalidation_tool", ROOT / "tools/validate_undersampling_common_case.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "UEE_phase.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="Pinned reference mismatch"):
        module.load_reference(tmp_path)
