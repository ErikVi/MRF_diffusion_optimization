"""sequence / bspline. Numerical behavior retained from the validated baseline."""

import jax
import jax.numpy as jnp
from jax import lax, jacobian
from functools import partial


def _basis_function(i, k, t, knots):
    if k == 0:
        return jnp.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)
    else:
        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]
        term1 = jnp.where(
            denom1 != 0,
            (t - knots[i]) / denom1 * _basis_function(i, k - 1, t, knots),
            0.0,
        )
        term2 = jnp.where(
            denom2 != 0,
            (knots[i + k + 1] - t) / denom2 * _basis_function(i + 1, k - 1, t, knots),
            0.0,
        )
        return term1 + term2


@partial(jax.jit, static_argnames=["degree"])
def evaluate_bspline(t_values, knots, coefficients, degree=3):
    """Evaluate a degree-k spline at dimensionless sample coordinates.

    Coefficient count is len(knots)-degree-1. The exact final knot is excluded
    (half-open intervals). Output has len(t_values) entries; degree is static."""

    def _basis_function(i, k, t, knots):
        if k == 0:
            return jnp.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)
        else:
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]
            term1 = jnp.where(
                denom1 != 0,
                (t - knots[i]) / denom1 * _basis_function(i, k - 1, t, knots),
                0.0,
            )
            term2 = jnp.where(
                denom2 != 0,
                (knots[i + k + 1] - t)
                / denom2
                * _basis_function(i + 1, k - 1, t, knots),
                0.0,
            )
            return term1 + term2

    def scan_fn(curve_points, i):
        v = _basis_function(i, degree, t_values, knots)
        return (curve_points + v * coefficients[i], None)

    curve_points = jnp.zeros(len(t_values))
    curve_points, _ = lax.scan(scan_fn, curve_points, jnp.arange(len(coefficients)))
    return curve_points


def fit_bspline_coefficients(y_values, knots, degree):
    """Fit coefficients by least squares on linspace(0,N,N), N=len(y_values).

    This historical grid differs from arange(N) in experiment reconstruction
    (B01). Uses the same basis as evaluation; output count len(knots)-degree-1.
    Host routine: Python loops and non-static degree preclude whole-function JIT."""
    num_coeffs = len(knots) - degree - 1
    num_points = len(y_values)
    t_values = jnp.linspace(0, num_points, num_points)
    B = jnp.zeros((num_points, num_coeffs))
    for i in range(num_coeffs):
        B = B.at[:, i].set(_basis_function(i, degree, t_values, knots))
    coefficients, _, _, _ = jax.numpy.linalg.lstsq(B, y_values, rcond=None)
    return coefficients


evaluate_bspline_matrix = evaluate_bspline
evaluate_bspline_vmap = evaluate_bspline
