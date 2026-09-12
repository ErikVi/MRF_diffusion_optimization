"information / jacobian. Numerical behavior retained from the validated baseline."

from mrf_diffusion.sequence.definition import LEGACY_SEQUENCE
import jax
import jax.numpy as jnp
from jax import lax, jacobian
from functools import partial
from mrf_diffusion.diffusion.metrics import legacy_fractional_anisotropy_gradient
from mrf_diffusion.diffusion.metrics import legacy_mean_diffusivity_gradient
from mrf_diffusion.diffusion.metrics import metric_matrix_pseudoinverse
from mrf_diffusion.simulation.signal import simulate_scalar_signal
from mrf_diffusion.simulation.signal import simulate_tensor_signal


@partial(jax.jit, static_argnums=(6,))
def scalar_signal_jacobian(
    flip_angle_train,
    phase_train,
    t1_ms,
    t2_ms,
    diffusivity,
    equilibrium_magnetization,
    sequence=LEGACY_SEQUENCE,
):
    """Return signal derivatives with shape (4,2,N), ordered [T1,T2,D,M].

    Uses JAX AD of simulate_scalar_signal; derivative units follow its ms,
    mm^2/s and normalized-magnetization parameters. Sequence is static."""
    return jnp.array(
        jacobian(simulate_scalar_signal, argnums=[2, 3, 4, 5])(
            flip_angle_train,
            phase_train,
            t1_ms,
            t2_ms,
            diffusivity,
            equilibrium_magnetization,
            sequence=sequence,
        )
    )


@partial(jax.jit, static_argnums=(8, 9, 10, 11, 12, 13, 14))
def legacy_tensor_metric_signal_jacobian(
    flip_angle_train,
    phase_train,
    t1_ms,
    t2_ms,
    equilibrium_magnetization,
    diffusion_tensor,
    preparation_flip_angles,
    preparation_phases,
    sampling=False,
    sampling_offset=0,
    sampling_rate=32,
    direction_count=3,
    state_count=20,
    include_inversion=False,
    sequence=LEGACY_SEQUENCE,
):
    """Return (5,2,samples), nominal order [T1,T2,M,FA,MD].

    First three derivatives use AD of simulate_tensor_signal. Last two
    contract the tensor signal derivative with matrix pseudoinverses of
    historical metric gradients. This does NOT establish a unique inverse
    tensor parameterization; metric-gradient defects remain (M01-M03).
    See docs/validation.md before interpreting diffusion information."""
    jacobian_scalars = jnp.array(
        jacobian(simulate_tensor_signal, argnums=[2, 3, 4])(
            flip_angle_train,
            phase_train,
            t1_ms,
            t2_ms,
            equilibrium_magnetization,
            diffusion_tensor,
            preparation_flip_angles,
            preparation_phases,
            sampling,
            sampling_offset,
            sampling_rate,
            direction_count,
            state_count,
            include_inversion,
            sequence=sequence,
        )
    )
    jacobian_diffusion = jnp.array(
        jacobian(simulate_tensor_signal, argnums=[5])(
            flip_angle_train,
            phase_train,
            t1_ms,
            t2_ms,
            equilibrium_magnetization,
            diffusion_tensor,
            preparation_flip_angles,
            preparation_phases,
            sampling,
            sampling_offset,
            sampling_rate,
            direction_count,
            state_count,
            include_inversion,
            sequence=sequence,
        )
    )
    anisotropy_jacobian = jnp.sum(
        jacobian_diffusion
        * metric_matrix_pseudoinverse(
            legacy_fractional_anisotropy_gradient(diffusion_tensor)
        ),
        axis=(-2, -1),
    )
    legacy_diffusivity_jacobian = jnp.sum(
        jacobian_diffusion
        * metric_matrix_pseudoinverse(legacy_mean_diffusivity_gradient()),
        axis=(-2, -1),
    )
    jacobian_combined = jnp.concatenate(
        [jacobian_scalars, anisotropy_jacobian, legacy_diffusivity_jacobian], axis=0
    )
    return jacobian_combined
