"""Constrained execution only: no physics, plotting, file IO or global counters."""

from scipy.optimize import minimize
import jax
from .settings import SolverSettings, ConstraintSettings


def optimize_sequence(
    objective,
    initial_coefficients,
    constraints,
    settings=SolverSettings(),
    bounds_settings=ConstraintSettings(),
    precision_evaluator=None,
):
    history = []
    previous_value = 0.0

    def callback(coefficients):
        nonlocal previous_value
        value = float(objective(coefficients))
        entry = {"iteration": len(history) + 1, "objective": value}
        if precision_evaluator is not None:
            entry["precision"] = precision_evaluator(coefficients).tolist()
        history.append(entry)
        if (
            len(history) > settings.minimum_callback_iterations
            and abs(value - previous_value) < settings.callback_change_tolerance
        ):
            raise StopIteration
        previous_value = value

    result = minimize(
        objective,
        initial_coefficients,
        jac=jax.jacobian(objective),
        method=settings.method,
        bounds=[
            (bounds_settings.coefficient_minimum, bounds_settings.coefficient_maximum)
        ]
        * len(initial_coefficients),
        constraints=constraints,
        options={"maxiter": settings.max_iterations},
        callback=callback,
    )
    return result, history
