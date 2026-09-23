"""Controlled phase search must not refit angles or lose valid candidates."""

import numpy as np
import jax.numpy as jnp
import pytest
from mrf_diffusion.sequence.parameterization import (
    make_legacy_knots,
    decode_sequence_parameters,
)
from mrf_diffusion.sequence.bspline import fit_bspline_coefficients
from mrf_diffusion.sequence.phase import generate_phase_train
from mrf_diffusion.optimization.phase_search import (
    phase_candidate,
    search_phase_fractions,
)


@pytest.mark.parametrize(
    "family", ["none", "quadratic", "linear", "sinusoidal", "alternating"]
)
def test_phase_candidates_preserve_angles_and_existing_projection(family):
    knots = make_legacy_knots(12, 8)
    points = jnp.arange(12.0)
    angles = jnp.array([0.2, 0.3, 0.4, 0.3, 0.2])
    coefficients, raw, applied = phase_candidate(angles, points, knots, family, 0.31)
    np.testing.assert_array_equal(coefficients[:5], angles)
    decoded, _ = decode_sequence_parameters(
        coefficients, points, knots, method="free form"
    )
    expected_angles, _ = decode_sequence_parameters(
        angles, points, knots, method="no phase modulation"
    )
    np.testing.assert_array_equal(decoded, expected_angles)
    expected_raw = (
        jnp.zeros(12)
        if family == "none"
        else generate_phase_train(points, 0.31, family)
    )
    np.testing.assert_array_equal(raw, expected_raw)
    expected_coefficients = fit_bspline_coefficients(expected_raw, knots, 3)
    np.testing.assert_allclose(coefficients[5:], expected_coefficients, rtol=0, atol=0)
    assert np.all(np.isfinite(applied))


def test_search_minimum_and_deterministic_ties():
    knots = make_legacy_knots(12, 8)
    args = (jnp.full(5, 0.3), jnp.arange(12.0), knots, "sinusoidal", [0.1, 0.2, 0.3])
    coefficients, best, rows = search_phase_fractions(
        *args, lambda c: jnp.sum(c[5:] ** 2)
    )
    assert best["objective"] == min(r["objective"] for r in rows)
    coefficients, best, rows = search_phase_fractions(*args, lambda c: 1.0)
    assert best["fraction"] == 0.1
    assert len(rows) == 3


def test_search_rejects_invalid_names_dimensions_and_nonfinite_objectives():
    args = (jnp.full(5, 0.3), jnp.arange(12.0), make_legacy_knots(12, 8))
    for family, fraction in [("misspelled", 0.1), ("linear", np.nan), ("linear", 1.1)]:
        with pytest.raises(ValueError):
            phase_candidate(*args, family, fraction)
    with pytest.raises(ValueError, match="coefficient"):
        phase_candidate(jnp.zeros(12), args[1], args[2], "none", 0.0)
    with pytest.raises(ValueError, match="No finite"):
        search_phase_fractions(*args, "linear", [0.1], lambda c: np.nan)
