"""epg / diffusion. Numerical behavior retained from the validated baseline."""

import jax
import jax.numpy as jnp
from jax import lax, jacobian
from functools import partial
from mrf_diffusion.epg.gradients import apply_negative_gradient_shift
from mrf_diffusion.epg.gradients import apply_positive_gradient_shift


@partial(jax.jit, static_argnums=(6, 7, 8, 9))
def relax_diffuse_scalar_and_shift(
    epg_states,
    t1_ms,
    t2_ms,
    duration_ms,
    diffusivity,
    equilibrium_magnetization,
    wavevector_step=10,
    order_step=1,
    gradient_enabled=1,
    truncate=1,
):
    """Apply scalar diffusion and relaxation, then optional EPG shift.

    epg_states has shape (3,K), rows F+, F-, Z. Times are ms, diffusivity
    is mm^2/s. wavevector_step is interpreted as angular wavevector/mm
    in analytical tests; experiment calibration remains unresolved.
    For unit gradients b+/-=(n^2 +/- n + 1/3)*k^2*duration_s;
    bZ=n^2*k^2*duration_s. Attenuation is exp(-b*diffusivity).
    See docs/scientific_model.md for controls and preserved limitations."""
    E1 = jnp.exp(-duration_ms / t1_ms)
    E2 = jnp.exp(-duration_ms / t2_ms)
    EE = jnp.diag(jnp.array([E2, E2, E1]))
    E1a = (
        jnp.zeros(epg_states.shape[1]).at[0].set(equilibrium_magnetization * (1.0 - E1))
    )
    state_orders = jnp.arange(epg_states.shape[1], dtype=jnp.float32)
    bvalZ = (state_orders * wavevector_step) ** 2 * duration_ms / 1000
    bvalp = (
        ((state_orders + 0.5 * gradient_enabled) * wavevector_step) ** 2
        * duration_ms
        / 1000
        + gradient_enabled * wavevector_step**2 / 12 * duration_ms / 1000
    )
    bvalm = (
        ((-state_orders + 0.5 * gradient_enabled) * wavevector_step) ** 2
        * duration_ms
        / 1000
        + gradient_enabled * wavevector_step**2 / 12 * duration_ms / 1000
    )
    epg_states = jnp.stack(
        [
            epg_states[0, :] * E2 * jnp.exp(-bvalp * diffusivity),
            epg_states[1, :] * E2 * jnp.exp(-bvalm * diffusivity),
            (epg_states[2, :] * E1 + E1a) * jnp.exp(-bvalZ * diffusivity),
        ]
    )
    if gradient_enabled == 1.0:
        epg_states = lax.cond(
            order_step >= 0.0,
            lambda x: apply_positive_gradient_shift(x, order_step, truncate),
            lambda x: apply_negative_gradient_shift(x, order_step, truncate),
            epg_states,
        )
    return epg_states


@partial(jax.jit, static_argnums=(7, 8, 9, 10))
def relax_diffuse_tensor_and_shift(
    epg_states,
    t1_ms,
    t2_ms,
    duration_ms,
    diffusion_tensor,
    equilibrium_magnetization,
    gradient_direction,
    wavevector_step=10,
    order_step=1,
    gradient_enabled=1,
    truncate=1,
):
    """Historical tensor diffusion/relaxation on (3,K) EPG states.

    diffusion_tensor is (3,3) in mm^2/s; times are ms; gradient_direction
    is (3,) and is NOT normalized here. The b tensor is contracted with D.
    Known defects D01/D02: cross terms cancel, F+ and F- share attenuation,
    and gradient_enabled=0 still permits nonzero dk. These are preserved,
    not accepted as valid anisotropic EPG physics. See docs/validation.md."""
    E1 = jnp.exp(-duration_ms / t1_ms)
    E2 = jnp.exp(-duration_ms / t2_ms)
    EE = jnp.diag(jnp.array([E2, E2, E1]))
    duration_ms = duration_ms / 1000
    E1a = (
        jnp.zeros(epg_states.shape[1]).at[0].set(equilibrium_magnetization * (1.0 - E1))
    )
    state_orders = jnp.arange(epg_states.shape[1], dtype=jnp.float32)
    k1 = jnp.outer(gradient_direction, state_orders * wavevector_step)
    k2 = jnp.outer(gradient_direction, (state_orders + order_step) * wavevector_step)
    b_s_L = jnp.einsum("in,jn->ijn", k1, k1) * duration_ms
    cross1 = jnp.einsum("in,jn->ijn", k1, k2)
    cross2 = jnp.einsum("in,jn->ijn", k1, k2)
    cross_term = 0.5 * (cross1 - cross2) * duration_ms
    dk = k2 - k1
    pure_term = jnp.einsum("in,jn->ijn", dk, dk) * (duration_ms / 3.0)
    b_s_T = b_s_L + cross_term + pure_term
    diffusion_decay_T = jnp.exp(-jnp.einsum("mnr,mn->r", b_s_T, diffusion_tensor))
    diffusion_decay_L = jnp.exp(-jnp.einsum("mnr,mn->r", b_s_L, diffusion_tensor))
    epg_states = jnp.stack(
        [
            epg_states[0, :] * E2 * diffusion_decay_T,
            epg_states[1, :] * E2 * diffusion_decay_T,
            (epg_states[2, :] * E1 + E1a) * diffusion_decay_L,
        ]
    )
    if gradient_enabled == 1.0:
        epg_states = lax.cond(
            order_step >= 0,
            lambda x: apply_positive_gradient_shift(x, order_step, truncate),
            lambda x: apply_negative_gradient_shift(x, order_step, truncate),
            epg_states,
        )
    return epg_states
