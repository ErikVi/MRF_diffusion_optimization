"""diffusion / metrics. Numerical behavior retained from the validated baseline."""

import jax
import jax.numpy as jnp


@jax.jit
def fractional_anisotropy(diffusion_tensor):
    """Return dimensionless FA for a symmetric nonzero (3,3) tensor.

    R=D/trace(D); FA=sqrt((3-1/trace(R@R))/2). No PSD validation or zero-tensor
    special case is applied. AD is tested away from isotropy; zero is undefined."""
    trace_D = jnp.trace(diffusion_tensor)
    R = diffusion_tensor / trace_D
    R_squared = jnp.dot(R, R)
    trace_R_squared = jnp.trace(R_squared)
    anisotropy = jnp.sqrt(0.5 * (3 - 1 / trace_R_squared))
    return anisotropy


@jax.jit
def legacy_fractional_anisotropy_gradient(diffusion_tensor):
    """Return the historical (3,3) claimed FA derivative.

    This does not agree with AD of fractional_anisotropy (M03).
    It remains only to preserve the legacy information calculation."""
    trace_D = jnp.trace(diffusion_tensor)
    R = diffusion_tensor / trace_D
    R_squared = jnp.dot(R, R)
    trace_R_squared = jnp.trace(R_squared)
    anisotropy = fractional_anisotropy(diffusion_tensor)
    dFA_dTr_R2 = -anisotropy / (2 * trace_R_squared**2)
    dTr_R2_dR = 2 * R
    dFA_dR = dFA_dTr_R2 * dTr_R2_dR
    I = jnp.eye(3)
    dR_dD = I / trace_D - diffusion_tensor / trace_D**2
    legacy_fractional_anisotropy_gradient = jnp.dot(dFA_dR, dR_dD)
    return legacy_fractional_anisotropy_gradient


@jax.jit
def legacy_mean_diffusivity(diffusion_tensor):
    """Return trace(D), NOT physical MD=trace(D)/3 (M01).

    Preserves the historical scalar result in mm^2/s for a (3,3) tensor."""
    return jnp.mean(jnp.trace(diffusion_tensor))


@jax.jit
def legacy_mean_diffusivity_gradient():
    """Return I/3, inconsistent with legacy_mean_diffusivity's derivative (M02)."""
    return jnp.diag(jnp.full(3, 1 / 3))


@jax.jit
def metric_matrix_pseudoinverse(matrix):
    """Return jnp.linalg.pinv(matrix); this is not an inverse tensor parameterization."""
    return jnp.linalg.pinv(matrix)
