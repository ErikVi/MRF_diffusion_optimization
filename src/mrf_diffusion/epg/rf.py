"""epg / rf. Numerical behavior retained from the validated baseline."""

import jax
import jax.numpy as jnp


@jax.jit
def apply_rf_rotation(epg_states, flip_angle, rf_phase=-jnp.pi / 2):
    """Rotate all (3,K) EPG states by an instantaneous RF pulse.

    Rows are F+, F-, Z. Flip angle and RF phase are radians; phase zero
    sends +Z to -imaginary F+ for a positive pi/2 pulse. Returns (3,K).
    RF default phase is -pi/2; simulators supply phase explicitly.
    See docs/conventions.md for the matrix and literature references."""
    exp_2j_phi = jnp.exp(2j * rf_phase)
    exp_j_phi = jnp.exp(1j * rf_phase)
    row_0 = jnp.array(
        [
            jnp.cos(flip_angle / 2) ** 2,
            exp_2j_phi * jnp.sin(flip_angle / 2) ** 2,
            -1j * exp_j_phi * jnp.sin(flip_angle),
        ]
    )
    row_1 = jnp.array(
        [
            jnp.conj(exp_2j_phi) * jnp.sin(flip_angle / 2) ** 2,
            jnp.cos(flip_angle / 2) ** 2,
            1j * jnp.conj(exp_j_phi) * jnp.sin(flip_angle),
        ]
    )
    row_2 = jnp.array(
        [
            -1j / 2 * jnp.conj(exp_j_phi) * jnp.sin(flip_angle),
            1j / 2 * exp_j_phi * jnp.sin(flip_angle),
            jnp.cos(flip_angle),
        ]
    )
    rotation_matrix = jnp.stack([row_0, row_1, row_2], axis=0)
    epg_states = jnp.matmul(rotation_matrix, epg_states)
    return epg_states
