"""epg / gradients. Numerical behavior retained from the validated baseline."""

import jax
import jax.numpy as jnp
from jax import lax, jacobian
from functools import partial


@partial(jax.jit, static_argnums=(1, 2))
def apply_positive_gradient_shift(epg_states, order_step=1, truncate=1):
    """Shift F+ toward higher order and F- toward lower order.

    Input/output rows are F+, F-, Z. Unit positive shifts enforce
    F+0=conj(F-0) and discard the departing highest F- order.
    truncate=0 adds exactly ONE empty column, independent of order_step;
    otherwise shape (3,K) is retained. Only unit/zero steps are validated."""
    if truncate == 0.0:
        epg_states = jnp.concatenate(
            (epg_states, jnp.array([[0.0], [0.0], [0.0]])), axis=1
        )
    N = epg_states.shape[1]
    shifted_positive_states = jnp.concatenate(
        [epg_states[:, -order_step:], epg_states[:, :-order_step]], axis=1
    )
    shifted_negative_states = jnp.concatenate(
        [epg_states[:, order_step:], epg_states[:, :order_step]], axis=1
    )
    epg_states = epg_states.at[0:2, :].set(
        jnp.stack([shifted_positive_states[0, :], shifted_negative_states[1, :]])
    )
    zero_block = jnp.zeros((1, order_step), dtype=epg_states.dtype)
    epg_states = lax.dynamic_update_slice(epg_states, zero_block, (1, N - order_step))
    epg_states = epg_states.at[0, 0].set(jnp.conj(epg_states[1, 0]))
    return epg_states


@partial(jax.jit, static_argnums=(1, 2))
def apply_negative_gradient_shift(epg_states, order_step=1, truncate=1):
    """Historical negative shift on (3,K) EPG states.

    order_step is a nonnegative magnitude when called directly; truncate=0
    adds one column. The zero-order boundary is WRONG for arriving F+1
    (G01); preserved for numerical continuity. See docs/validation.md."""
    if truncate == 0.0:
        epg_states = jnp.concatenate(
            (epg_states, jnp.array([[0.0], [0.0], [0.0]])), axis=1
        )
    N = epg_states.shape[1]
    shifted_positive_states = jnp.concatenate(
        [epg_states[:, order_step:], epg_states[:, :order_step]], axis=1
    )
    shifted_negative_states = jnp.concatenate(
        [epg_states[:, -order_step:], epg_states[:, :-order_step]], axis=1
    )
    epg_states = epg_states.at[0:2, :].set(
        jnp.stack([shifted_positive_states[0, :], shifted_negative_states[1, :]])
    )
    zero_block = jnp.zeros((1, order_step), dtype=epg_states.dtype)
    epg_states = lax.dynamic_update_slice(epg_states, zero_block, (0, N - order_step))
    epg_states = epg_states.at[0, 0].set(jnp.conj(epg_states[1, 0]))
    return epg_states
