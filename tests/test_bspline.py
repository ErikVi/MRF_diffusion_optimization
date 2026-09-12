"""SciPy provides an independent reference on the spline's base interval."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import BSpline
import EPG_blocks_jaxcode as epg
from tests.cases import KNOTS, COEFFICIENTS, POINTS

EVALUATORS = [epg.bspline, epg.bspline_vectorized, epg.bspline_vmap]


@pytest.mark.parametrize("evaluate", EVALUATORS)
def test_basis_partition_and_scipy_consistency(evaluate):
    basis = np.stack([evaluate(POINTS, KNOTS, c) for c in jnp.eye(5)], axis=1)
    expected = BSpline.design_matrix(np.asarray(POINTS), np.asarray(KNOTS), 3).toarray()
    assert basis.shape == (8, 5)
    assert_allclose(basis, expected, atol=1e-14)
    assert_allclose(basis.sum(axis=1), 1., atol=1e-14)
    assert np.all(basis >= 0.)
    assert_allclose(evaluate(POINTS, KNOTS, COEFFICIENTS), expected@COEFFICIENTS, atol=1e-14)


def test_fit_coefficient_count_and_reconstruction_on_actual_fit_grid():
    # Legacy fitting API uses [0,N], not [0,N-1]. Test its actual contract separately.
    knots = KNOTS.at[5:].set(8.01)
    fit_points = np.linspace(0, 8, 8)
    samples = BSpline(np.asarray(knots), np.asarray(COEFFICIENTS), 3)(fit_points)
    fitted = epg.bspline_coefficients(jnp.asarray(samples), knots, 3)
    assert fitted.shape == (len(knots)-3-1,)
    assert_allclose(fitted, COEFFICIENTS, atol=1e-12)
    assert_allclose(epg.bspline(jnp.asarray(fit_points), knots, fitted), samples, atol=1e-12)


@pytest.mark.discrepancy
@pytest.mark.xfail(reason="B01: experiment reconstruction grid differs from fit grid", raises=AssertionError)
def test_fit_reconstructs_on_experiment_sample_grid():
    knots = KNOTS.at[5:].set(8.01)
    samples = BSpline(np.asarray(knots), np.asarray(COEFFICIENTS), 3)(np.asarray(POINTS))
    fitted = epg.bspline_coefficients(jnp.asarray(samples), knots, 3)
    assert_allclose(epg.bspline(POINTS, knots, fitted), samples, atol=1e-12)


@pytest.mark.parametrize("evaluate", EVALUATORS)
def test_left_boundary_and_right_hand_limit(evaluate):
    points = jnp.array([0., 8.-1e-9])
    assert_allclose(evaluate(points, KNOTS, COEFFICIENTS),
                    BSpline(np.asarray(KNOTS), np.asarray(COEFFICIENTS), 3)(points), atol=1e-14)


@pytest.mark.parametrize("evaluate", EVALUATORS)
def test_legacy_exact_right_boundary_is_excluded(evaluate):
    # Characterization of the half-open basis, NOT a claim of closed-end correctness.
    assert_allclose(evaluate(jnp.array([8.]), KNOTS, COEFFICIENTS), [0.], atol=0)


def test_spline_coefficient_derivative_and_jit():
    derivative = jax.jacobian(lambda c: epg.bspline(POINTS, KNOTS, c))(COEFFICIENTS)
    expected = BSpline.design_matrix(np.asarray(POINTS), np.asarray(KNOTS), 3).toarray()
    assert_allclose(derivative, expected, atol=1e-14)
    compiled = epg.bspline(POINTS, KNOTS, COEFFICIENTS)
    with jax.disable_jit():
        assert_allclose(epg.bspline(POINTS, KNOTS, COEFFICIENTS), compiled, atol=1e-14)
