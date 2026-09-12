"""Original experiment coefficients, affine phase terms and constraint signs."""

import json
from pathlib import Path
import jax.numpy as jnp
from numpy.testing import assert_allclose
import pytest
from mrf_diffusion.sequence.parameterization import (
    initialize_sequence_parameters,
    make_legacy_knots,
)
from mrf_diffusion.optimization.constraints import make_sequence_constraints

REFERENCE = json.loads(
    (Path(__file__).parent / "reference/experiment_initialization.json").read_text()
)


@pytest.mark.parametrize("method", list(REFERENCE["cases"]))
def test_initial_coefficients_affine_phase_and_constraints_match_original(method):
    angles = jnp.asarray(REFERENCE["angles_rad"])
    points = jnp.arange(len(angles), dtype=float)
    knots = make_legacy_knots(len(angles), REFERENCE["knot_setting"])
    actual = initialize_sequence_parameters(angles, points, knots, method)
    expected = REFERENCE["cases"][method]
    for value, reference in zip(actual, expected["initial"]):
        assert_allclose(value, reference, rtol=1e-12, atol=1e-12)
    for constraint, reference in zip(
        make_sequence_constraints(points, knots, method), expected["constraints"]
    ):
        assert_allclose(constraint["fun"](actual[0]), reference, rtol=1e-12, atol=1e-12)
