"""Inverse-information calculations; singular inputs remain unregularized.

These operations intentionally preserve inv() semantics. They do not turn a
singular information matrix into a valid uncertainty estimate.
"""

import jax.numpy as jnp


def inverse_fisher_information(fisher_information):
    return jnp.linalg.inv(fisher_information)


def relative_standard_deviation_bounds(fisher_information, parameters):
    """sqrt(diag(F^-1)/theta^2), for the supplied estimated parameter set."""
    return jnp.sqrt(
        jnp.diag(inverse_fisher_information(fisher_information)) / parameters**2
    )


def information_diagnostics(fisher_information, parameter_scales):
    """Report scaled singular values/condition without altering the matrix."""
    scaled = fisher_information * jnp.outer(parameter_scales, parameter_scales)
    values = jnp.linalg.svd(scaled, compute_uv=False)
    return {"singular_values": values, "condition_number": values[0] / values[-1]}
