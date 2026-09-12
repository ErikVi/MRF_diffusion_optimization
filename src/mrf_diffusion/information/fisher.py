"information / fisher. Numerical behavior retained from the validated baseline."

from mrf_diffusion.sequence.definition import LEGACY_SEQUENCE
import jax
import jax.numpy as jnp
from functools import partial
from mrf_diffusion.information.jacobian import legacy_tensor_metric_signal_jacobian
from mrf_diffusion.information.jacobian import scalar_signal_jacobian


@partial(jax.jit, static_argnums=(6, 7))
def scalar_fisher_information(
    flip_angle_train,
    phase_train,
    t1_ms,
    t2_ms,
    diffusivity,
    equilibrium_magnetization,
    sigma=10 ** (-1.65),
    sequence=LEGACY_SEQUENCE,
):
    """Return (4,4) information in [T1,T2,D,M] order.

    Flatten the (4,2,N) real/imaginary Jacobian J and calculate J @ J.T/sigma^2.
    sigma is independent Gaussian SD per real channel, in signal units.
    No rank guard or regularization is applied. Sequence and sigma are static."""
    t1_ms = jnp.asarray(t1_ms, dtype=jnp.float64)
    t2_ms = jnp.asarray(t2_ms, dtype=jnp.float64)
    diffusivity = jnp.asarray(diffusivity, dtype=jnp.float64)
    equilibrium_magnetization = jnp.asarray(
        equilibrium_magnetization, dtype=jnp.float64
    )
    J = scalar_signal_jacobian(
        flip_angle_train,
        phase_train,
        t1_ms,
        t2_ms,
        diffusivity,
        equilibrium_magnetization,
        sequence=sequence,
    )
    J = J.reshape(J.shape[0], -1)
    FIM = J @ J.T
    return 1 / sigma**2 * FIM


@partial(jax.jit, static_argnums=(8, 9, 10, 11, 12, 13, 14, 15))
def legacy_tensor_fisher_information(
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
    sigma=10 ** (-1.65),
    sequence=LEGACY_SEQUENCE,
):
    """Return (5,5) J @ J.T/sigma^2 in nominal [T1,T2,M,FA,MD] order.

    Uses legacy_tensor_metric_signal_jacobian, including its invalid metric
    derivatives. sigma is per real/imaginary channel SD in signal units.
    The optimization layer selects its leading (3,3) block BEFORE inversion."""
    t1_ms = jnp.asarray(t1_ms, dtype=jnp.float64)
    t2_ms = jnp.asarray(t2_ms, dtype=jnp.float64)
    equilibrium_magnetization = jnp.asarray(
        equilibrium_magnetization, dtype=jnp.float64
    )
    diffusion_tensor = jnp.asarray(diffusion_tensor, dtype=jnp.float64)
    J = legacy_tensor_metric_signal_jacobian(
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
    J = J.reshape(J.shape[0], -1)
    FIM = J @ J.T
    return 1 / sigma**2 * FIM
