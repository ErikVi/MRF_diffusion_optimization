"""MSc inequality constraints; retain original ordering and derivative policy."""

import jax
import jax.numpy as jnp
from mrf_diffusion.sequence.bspline import evaluate_bspline
from mrf_diffusion.sequence.parameterization import angle_coefficients
from .settings import ConstraintSettings


def make_sequence_constraints(points, knots, method, settings=ConstraintSettings()):
    def angles(coefficients):
        return evaluate_bspline(points, knots, angle_coefficients(coefficients, method))

    def lower(coefficients):
        return angles(coefficients) - settings.minimum_flip_angle

    constraints = [
        {"type": "ineq", "fun": lower, "jac": jax.jacobian(lower)},
        {"type": "ineq", "fun": lambda c: settings.maximum_flip_angle - angles(c)},
        {
            "type": "ineq",
            "fun": lambda c: settings.maximum_angle_step - jnp.abs(jnp.diff(angles(c))),
        },
    ]
    if method == "quadratic malleable":
        constraints.extend(
            [
                {
                    "type": "ineq",
                    "fun": lambda c: c[-3] - settings.minimum_split_fraction,
                },
                {
                    "type": "ineq",
                    "fun": lambda c: settings.maximum_split_fraction - c[-3],
                },
                {
                    "type": "ineq",
                    "fun": lambda c: c[-2] - settings.minimum_positive_curvature,
                },
                {
                    "type": "ineq",
                    "fun": lambda c: settings.maximum_negative_curvature - c[-1],
                },
            ]
        )
    return constraints
