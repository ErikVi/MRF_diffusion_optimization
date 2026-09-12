"""epg / relaxation. Numerical behavior retained from the validated baseline."""

import jax
import jax.numpy as jnp
from functools import partial
from mrf_diffusion.epg.gradients import apply_positive_gradient_shift


@partial(jax.jit, static_argnums=(4, 5, 6))
def relax_and_shift(
    epg_states,
    t1_ms,
    t2_ms,
    duration_ms,
    order_step=1,
    gradient_enabled=1.0,
    truncate=1,
):
    """Apply exponential relaxation, Z0 recovery, and optional EPG shift.

    Input/output shape (3,K), rows F+, F-, Z. Time and T1/T2 are ms;
    equilibrium_magnetization is normalized magnetization. Recovery is
    M*(1-exp(-duration_ms/t1_ms)) into Z0 only. The decay matrix retains
    legacy float32 arithmetic. Gradient/truncation controls are static."""
    E2 = jnp.exp(-duration_ms / t2_ms)
    E1 = jnp.exp(-duration_ms / t1_ms)
    EE = jnp.diag(jnp.array([E2, E2, E1], dtype=jnp.float32))
    epg_states = jnp.matmul(EE, epg_states)
    epg_states = epg_states.at[2, 0].add(1.0 - E1)
    if gradient_enabled == 1.0:
        epg_states = apply_positive_gradient_shift(epg_states, order_step, truncate)
    return epg_states
