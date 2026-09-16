"""Information diagnostic regression: physical tensor perturbations remain finite."""

import importlib.util
from pathlib import Path
import numpy as np
from mrf_diffusion.sequence.definition import MRFSequence, SimulationOptions
from mrf_diffusion.simulation.api import simulate_mrf_signal
from mrf_diffusion.simulation.tissue import TissueParameters
from mrf_diffusion.diffusion.parameterization import axisymmetric_tensor


def test_physical_parameter_finite_difference_convergence(tmp_path):
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/end_to_end_validation/information_diagnostic.py"
    )
    spec = importlib.util.spec_from_file_location("e2e_information", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sequence = MRFSequence(np.linspace(0.25, 0.35, 12), 0.18 * np.arange(12) ** 2)
    result = module.evaluate(
        sequence, SimulationOptions(direction_count=3, state_count=16), tmp_path, "case"
    )
    data = np.load(tmp_path / "case_physical_parameter_information.npz")
    assert data["jacobian"].shape == (72, 6)
    assert result["derivative_step_convergence"] < 1e-5
    np.testing.assert_allclose(data["fim"], data["fim"].T, rtol=1e-12, atol=1e-12)
    assert np.linalg.eigvalsh(data["scaled_fim"]).min() > 0
    assert result["rank"] == 6
    components = np.asarray(
        simulate_mrf_signal(
            TissueParameters(
                750.0, 70.0, 1.0, axisymmetric_tensor(0.0007, 0.2, (1.0, 0.0, 0.0))
            ),
            sequence,
            SimulationOptions(direction_count=3, state_count=16),
        )
    )
    # Density is a whole-fingerprint multiplier, not the internal equilibrium M.
    np.testing.assert_allclose(
        data["jacobian"][:, 2], components.ravel(), rtol=1e-9, atol=1e-10
    )
    # A constant object phase rotates real/imaginary signal exactly once.
    np.testing.assert_allclose(
        data["jacobian"][:, 5],
        np.concatenate((-components[1], components[0])),
        rtol=1e-8,
        atol=1e-10,
    )
