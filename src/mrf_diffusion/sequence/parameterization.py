"""One sequence reconstruction for objectives, constraints and reporting."""

import jax.numpy as jnp
from .bspline import evaluate_bspline, fit_bspline_coefficients
from .phase import generate_phase_train, generate_piecewise_quadratic_phase


def angle_coefficients(coefficients, method):
    if method == "quadratic malleable":
        return coefficients[:-3]
    if method == "no phase modulation":
        return coefficients
    return coefficients[: len(coefficients) // 2]


def decode_sequence_parameters(
    coefficients,
    eval_points,
    knots,
    phase_offset=0.0,
    phase_slope=0.0,
    method="quadratic",
):
    """Decode the four historical methods, including their discrete phase split."""
    if method == "quadratic":
        curvature = evaluate_bspline(
            eval_points, knots, coefficients[len(coefficients) // 2 :]
        )
        phase_train = (
            jnp.cumsum(jnp.cumsum(curvature)) + phase_slope * eval_points + phase_offset
        )
    elif method == "free form":
        phase_train = evaluate_bspline(
            eval_points, knots, coefficients[len(coefficients) // 2 :]
        )
    elif method == "no phase modulation":
        phase_train = jnp.zeros(len(eval_points))
    elif method == "quadratic malleable":
        split_fraction, left_curvature, right_curvature = coefficients[-3:]
        phase_train = generate_piecewise_quadratic_phase(
            eval_points,
            split_fraction,
            left_curvature,
            right_curvature,
            discontinuity_type="nulling",
        )
    else:
        raise ValueError(f"Unknown phase method: {method}")
    flip_angles = evaluate_bspline(
        eval_points, knots, angle_coefficients(coefficients, method)
    )
    return flip_angles, phase_train


def make_legacy_knots(sample_count, knot_setting, degree=3):
    """Preserve N+.01 endpoint and legacy 'knot setting' (not coefficient count)."""
    return jnp.concatenate(
        (
            jnp.zeros(degree),
            jnp.linspace(0.0, sample_count + 0.01, knot_setting + 1 - 2 * degree),
            jnp.ones(degree) * (sample_count + 0.01),
        )
    )


def initialize_sequence_parameters(
    flip_angles,
    points,
    knots,
    method,
    degree=3,
    phase_fraction=0.18,
    initial_phase_method="quadratic",
    piecewise_parameters=(0.3, 0.04, -0.03),
):
    """Exact MSc initializer, retaining the fit/reconstruction coordinate discrepancy."""
    angle_coeffs = fit_bspline_coefficients(flip_angles, knots, degree)
    if method == "no phase modulation":
        return angle_coeffs, jnp.zeros(len(points)), 0.0, 0.0
    if method == "quadratic malleable":
        phase_coeffs = jnp.asarray(piecewise_parameters)
        phase = generate_piecewise_quadratic_phase(
            points, *phase_coeffs, discontinuity_type="nulling"
        )
        return jnp.concatenate((angle_coeffs, phase_coeffs)), phase, 0.0, 0.0
    phase = generate_phase_train(flip_angles, phase_fraction, initial_phase_method)
    if method == "free form":
        phase_coeffs = fit_bspline_coefficients(phase, knots, degree)
        return jnp.concatenate((angle_coeffs, phase_coeffs)), phase, 0.0, 0.0
    if method != "quadratic":
        raise ValueError(f"Unknown phase method: {method}")
    phase_coeffs = fit_bspline_coefficients(jnp.diff(jnp.diff(phase)), knots, degree)
    integrated = jnp.cumsum(jnp.cumsum(evaluate_bspline(points, knots, phase_coeffs)))
    indices = jnp.arange(len(integrated))
    design = jnp.stack([indices, jnp.ones_like(indices)], axis=1)
    slope, offset = jnp.linalg.lstsq(design, phase - integrated, rcond=None)[0]
    return jnp.concatenate((angle_coeffs, phase_coeffs)), phase, offset, slope
