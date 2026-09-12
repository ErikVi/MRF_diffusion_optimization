"""Pre-refactor signals, objectives and gradients, never recaptured by tests."""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from mrf_diffusion.simulation.signal import simulate_tensor_signal
from mrf_diffusion.optimization.objectives import tensor_ensemble_objective
from mrf_diffusion.optimization.evaluation import tensor_sequence_precision_bounds
from tests.cases import ANGLES, PHASES, TENSOR, PREP

REFERENCE = json.loads(
    (Path(__file__).parent / "reference/architecture_objectives.json").read_text()
)["cases"]


def check_reference(name, value):
    expected = np.asarray(REFERENCE[name]["values"])
    assert value.shape == expected.shape
    assert_allclose(value, expected, rtol=2e-7, atol=1e-9)


@pytest.mark.parametrize("inversion", [False, True])
def test_direction_blocks_and_inversion_reference(inversion):
    value = simulate_tensor_signal(
        ANGLES[:4],
        PHASES[:4],
        1000.0,
        80.0,
        0.9,
        TENSOR,
        PREP,
        PREP,
        False,
        0,
        32,
        2,
        5,
        inversion,
    )
    check_reference("tensor_signal_inversion_" + str(inversion), value)


@pytest.mark.parametrize(
    "method", ["free form", "no phase modulation", "quadratic", "quadratic malleable"]
)
def test_phase_method_bounds_objectives_and_gradient_reference(method):
    knots = jnp.array([0.0, 0.0, 0.0, 0.0, 6.01, 6.01, 6.01, 6.01])
    points = jnp.arange(6.0)
    angles = jnp.array([0.2, 0.4, 0.6, 0.3])
    phase = jnp.array([0.03, -0.02, 0.08, 0.02])
    coeff = (
        angles
        if method == "no phase modulation"
        else jnp.concatenate(
            [
                angles,
                (
                    jnp.array([0.4, 0.04, -0.03])
                    if method == "quadratic malleable"
                    else phase
                ),
            ]
        )
    )
    args = (
        coeff,
        points,
        knots,
        jnp.array([[1000.0, 80.0, 0.9]]),
        TENSOR[None],
        jnp.ones(5),
        jnp.ones(1),
        0.1,
        0.02,
        PREP,
        PREP,
        False,
        0,
        32,
        1,
        5,
        False,
    )
    check_reference("bounds_" + method, tensor_sequence_precision_bounds(*args, method))
    for aggregation in ["L1", "L2"]:
        check_reference(
            aggregation + "_" + method,
            tensor_ensemble_objective(*args, aggregation, method),
        )
    if method == "free form":
        check_reference(
            "objective_gradient",
            jax.jacobian(tensor_ensemble_objective)(*args, "L1", method),
        )
