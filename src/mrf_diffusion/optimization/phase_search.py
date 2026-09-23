"""Finite-grid search of existing spline-projected historical phase families.

Fraction can be discontinuous: do not differentiate it. Objective is supplied
unchanged by the caller; flip-angle coefficients are retained exactly.
"""

import time
import numpy as np
import jax.numpy as jnp
from mrf_diffusion.sequence.phase import generate_phase_train
from mrf_diffusion.sequence.bspline import fit_bspline_coefficients
from mrf_diffusion.sequence.parameterization import decode_sequence_parameters

PHASE_FAMILIES = ("none", "quadratic", "linear", "sinusoidal", "alternating")


def phase_candidate(angle_coefficients, points, knots, family, fraction):
    """Paired coefficients, raw radians and spline-applied radians; fraction [0,1]."""
    if family not in PHASE_FAMILIES:
        raise ValueError(f"Unknown phase family: {family}")
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be finite in [0,1]")
    angles = jnp.asarray(angle_coefficients)
    if angles.ndim != 1 or len(angles) != len(knots) - 4:
        raise ValueError("Expected one cubic angle coefficient block")
    raw = (
        jnp.zeros(len(points))
        if family == "none"
        else generate_phase_train(jnp.zeros(len(points)), fraction, family)
    )
    phase_coefficients = fit_bspline_coefficients(raw, knots, 3)
    paired = jnp.concatenate((angles, phase_coefficients))
    _, applied = decode_sequence_parameters(paired, points, knots, method="free form")
    return paired, raw, applied


def search_phase_fractions(
    angle_coefficients, points, knots, family, fractions, objective
):
    """Best coefficients/row and all rows; first finite minimum wins ties.

    Nonfinite values are recorded and cannot win; exceptions are not hidden.
    """
    fractions = np.asarray(fractions, float)
    if fractions.ndim != 1 or not len(fractions):
        raise ValueError("Provide a nonempty fraction grid")
    rows, best, best_coefficients = [], None, None
    for fraction in fractions:
        start = time.perf_counter()
        coefficients, raw, applied = phase_candidate(
            angle_coefficients, points, knots, family, float(fraction)
        )
        value = float(objective(coefficients))
        finite = bool(np.isfinite(value))
        row = dict(
            family=family,
            fraction=float(fraction),
            objective=value if finite else None,
            status="finite" if finite else "nonfinite",
            runtime_seconds=time.perf_counter() - start,
            projection_rmse_rad=float(
                np.sqrt(np.mean((np.asarray(raw) - np.asarray(applied)) ** 2))
            ),
        )
        rows.append(row)
        if finite and (best is None or value < best["objective"]):
            best, best_coefficients = row, coefficients
    if best is None:
        raise ValueError("No finite phase candidate objective")
    return best_coefficients, best, rows
