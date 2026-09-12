"""Small deterministic inputs; milliseconds, radians and mm^2/s throughout."""

import jax.numpy as jnp
import numpy as np

ANGLES = jnp.array([0.21, 0.47, 0.32, 0.66, 0.28, 0.52, 0.73, 0.39])
PHASES = jnp.array([0.0, 0.13, -0.21, 0.37, 0.52, -0.17, 0.24, 0.61])
PARAMETERS = jnp.array([1000.0, 80.0, 0.0008, 0.9])
TENSOR = jnp.array(
    [[0.0015, 0.0001, 0.0], [0.0001, 0.0007, 5e-05], [0.0, 5e-05, 0.0003]]
)
PREP = jnp.zeros(3)
KNOTS = jnp.array([0.0, 0.0, 0.0, 0.0, 4.0, 8.0, 8.0, 8.0, 8.0])
COEFFICIENTS = jnp.array([0.2, 0.4, 0.6, 0.3, 0.5])
POINTS = jnp.arange(8, dtype=jnp.float64)


def equilibrium(n=5):
    return jnp.zeros((3, n), dtype=jnp.complex128).at[2, 0].set(1.0)


def populated_states():
    return jnp.array(
        [
            [0.2 + 0.1j, 0.3 - 0.2j, 0.1j, 0.0, 0.0],
            [0.2 - 0.1j, 0.1 + 0.4j, -0.2j, 0.0, 0.0],
            [0.7, 0.1 + 0.05j, 0.2, 0.0, 0.0],
        ],
        dtype=jnp.complex128,
    )


def central_difference(function, parameters, steps):
    parameters = np.asarray(parameters, dtype=float)
    columns = []
    for i, step in enumerate(steps):
        offset = np.zeros_like(parameters)
        offset[i] = step
        columns.append(
            (
                np.asarray(function(parameters + offset))
                - np.asarray(function(parameters - offset))
            )
            / (2 * step)
        )
    return np.stack(columns)
