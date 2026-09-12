"""sequence / phase. Numerical behavior retained from the validated baseline."""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


def generate_phase_train(flip_angle_train, fraction, method):
    """Generate a historical phase family in radians using only train length.

    This is host NumPy code, not an AD/JIT kernel. Degree-domain increments
    are +/-2*n for quadratic and +/-10 for linear; both leave a zero-valued
    gap around N*fraction. Sinusoidal amplitude is 1000 degrees; alternating
    increments are (-1)^n*1000*fraction degrees. Finally convert to radians.
    Supported method strings: quadratic, linear, sinusoidal, alternating.
    Unknown strings retain the historical zero train rather than raising."""
    N = len(flip_angle_train)
    phase_train = np.zeros(N)
    if method == "quadratic":
        for n in range(1, N):
            if n < N * fraction:
                phase_train[n] = phase_train[n - 1] + 2 * n
            elif n > N * fraction + 1:
                phase_train[n] = phase_train[n - 1] - 2 * n
    elif method == "linear":
        for n in range(1, N):
            if n < N * fraction:
                phase_train[n] = phase_train[n - 1] + 10
            elif n > N * fraction + 1:
                phase_train[n] = phase_train[n - 1] - 10
    elif method == "sinusoidal":
        phase_train = 1000 * np.sin(2 * np.pi * (np.arange(N) / N + fraction))
    elif method == "alternating":
        for n in range(1, N):
            phase_train[n] = phase_train[n - 1] + (-1) ** n * 1000 * fraction
    phase_train = np.deg2rad(phase_train)
    return jnp.array(phase_train)


@partial(jax.jit, static_argnums=4)
def generate_piecewise_quadratic_phase(
    flip_angle_train, fraction, slope1, slope2, discontinuity_type="nulling"
):
    """Return two quadratic phase segments in radians, using only train length.

    Split index is floor(N*fraction). slope1/slope2 are quadratic coefficients
    in radians per squared sample index, despite the historical argument names.
    The right segment restarts at zero. Both 'nulling' and 'continuing' yield
    zero at the split: the latter name does NOT enforce continuity with the
    left segment. The discrete split location is not smoothly differentiable.
    An unknown discontinuity_type retains the historical zero train."""
    N = len(flip_angle_train)
    phase_train = jnp.zeros(N)
    split_index = jnp.floor(N * fraction).astype(int)
    if discontinuity_type == "nulling":
        phase_train = jnp.where(
            jnp.arange(N) < split_index,
            slope1 * jnp.square(jnp.arange(N)),
            jnp.where(
                jnp.arange(N) > split_index,
                slope2 * jnp.square(jnp.arange(N) - split_index),
                0.0,
            ),
        )
    elif discontinuity_type == "continuing":
        phase_train = jnp.where(
            jnp.arange(N) < split_index,
            slope1 * jnp.square(jnp.arange(N)),
            slope2 * jnp.square(jnp.arange(N) - split_index),
        )
    return phase_train
