"""Object-oriented input boundary, with ordinary inspectable numerical kernels."""

import jax.numpy as jnp
from .signal import simulate_scalar_signal, simulate_tensor_signal
from .tissue import TissueParameters
from mrf_diffusion.sequence.definition import MRFSequence, SimulationOptions


def simulate_mrf_signal(
    tissue: TissueParameters,
    sequence: MRFSequence,
    options=SimulationOptions(),
    *,
    tensor=True,
):
    """Return (2,samples) real/imaginary signal. Low-level kernels remain differentiable.

    The wrapper is a host API; use kernel arguments directly for JAX transforms.
    """
    angles, phases = jnp.asarray(sequence.flip_angles), jnp.asarray(sequence.rf_phases)
    if angles.ndim != 1 or angles.shape != phases.shape:
        raise ValueError(
            "Flip angles and RF phases must be equal-length one-dimensional trains"
        )
    if not tensor:
        return simulate_scalar_signal(
            angles,
            phases,
            tissue.t1_ms,
            tissue.t2_ms,
            tissue.diffusion,
            tissue.equilibrium_magnetization,
            sequence=sequence.settings,
        )
    prep_angles = (
        jnp.zeros(3)
        if sequence.preparation_flip_angles is None
        else jnp.asarray(sequence.preparation_flip_angles)
    )
    prep_phases = (
        jnp.zeros(3)
        if sequence.preparation_phases is None
        else jnp.asarray(sequence.preparation_phases)
    )
    return simulate_tensor_signal(
        angles,
        phases,
        tissue.t1_ms,
        tissue.t2_ms,
        tissue.equilibrium_magnetization,
        tissue.diffusion,
        prep_angles,
        prep_phases,
        options.sampling,
        options.sampling_offset,
        options.sampling_rate,
        options.direction_count,
        options.state_count,
        options.include_inversion,
        sequence.settings,
    )
