"""EPG layout: component rows F+, F-, Z; nonnegative order columns."""

import jax.numpy as jnp


def initial_epg_state(state_count, longitudinal_magnetization=1.0):
    """Return complex128 (3,K) with only Z0 populated; preserves legacy policy."""
    return (
        jnp.zeros((3, state_count), dtype=jnp.complex128)
        .at[2, 0]
        .set(longitudinal_magnetization)
    )


def transverse_signal(epg_states):
    """Complex observed F+0; no magnitude conversion or receiver phase removal."""
    return epg_states[0, 0]
