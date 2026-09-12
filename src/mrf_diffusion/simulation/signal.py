"""simulation / signal. Numerical behavior retained from the validated baseline."""

from mrf_diffusion.epg.states import initial_epg_state
from mrf_diffusion.sequence.definition import LEGACY_SEQUENCE
import jax
import jax.numpy as jnp
from jax import lax, jacobian
from functools import partial
from mrf_diffusion.epg.diffusion import relax_diffuse_scalar_and_shift
from mrf_diffusion.epg.diffusion import relax_diffuse_tensor_and_shift
from mrf_diffusion.epg.rf import apply_rf_rotation


@partial(jax.jit, static_argnums=(6,))
def simulate_scalar_signal(
    flip_angle_train,
    phase_train,
    t1_ms,
    t2_ms,
    diffusivity,
    equilibrium_magnetization,
    sequence=LEGACY_SEQUENCE,
):
    """Simulate a prepared scalar-diffusion MRF train, returning (2,N).

    Equal-length flip_angle_train and phase_train are radians; rows of the
    result are real and imaginary F+0 after TE. T1/T2 and timing are ms,
    diffusivity mm^2/s, M normalized. SequenceSettings contains explicit
    preparation/timing/order defaults. Readout RF inputs retain float32.
    Initial Z0 is sequence.initial_longitudinal_magnetization, not M."""
    echo_time_ms, repetition_time_ms, order_step = (
        sequence.echo_time_ms,
        sequence.repetition_time_ms,
        sequence.order_step,
    )
    state_count = sequence.scalar_state_count
    epg_states = initial_epg_state(
        state_count, sequence.initial_longitudinal_magnetization
    )
    epg_states = apply_rf_rotation(
        epg_states,
        flip_angle=sequence.preparation_angles[0],
        rf_phase=sequence.preparation_rf_phase,
    )
    epg_states = relax_diffuse_scalar_and_shift(
        epg_states,
        t1_ms,
        t2_ms,
        echo_time_ms,
        diffusivity,
        equilibrium_magnetization,
        wavevector_step=sequence.scalar_preparation_wavevector,
        order_step=1,
        gradient_enabled=1,
        truncate=1,
    )
    epg_states = apply_rf_rotation(
        epg_states,
        flip_angle=sequence.preparation_angles[1],
        rf_phase=sequence.preparation_rf_phase,
    )
    epg_states = relax_diffuse_scalar_and_shift(
        epg_states,
        t1_ms,
        t2_ms,
        echo_time_ms,
        diffusivity,
        equilibrium_magnetization,
        wavevector_step=sequence.scalar_preparation_wavevector,
        order_step=1,
        gradient_enabled=1,
        truncate=1,
    )
    epg_states = apply_rf_rotation(
        epg_states,
        flip_angle=sequence.preparation_angles[2],
        rf_phase=sequence.preparation_rf_phase,
    )

    def scan_fn(epg_states, inputs):
        flip_angle, rf_phase = inputs
        epg_states = apply_rf_rotation(
            epg_states, flip_angle=flip_angle, rf_phase=rf_phase
        )
        epg_states = relax_diffuse_scalar_and_shift(
            epg_states,
            t1_ms,
            t2_ms,
            echo_time_ms,
            diffusivity,
            equilibrium_magnetization,
            wavevector_step=sequence.scalar_readout_wavevector,
            order_step=order_step,
            gradient_enabled=0,
            truncate=1,
        )
        real_part = jnp.real(epg_states[0, 0])
        imag_part = jnp.imag(epg_states[0, 0])
        epg_states = relax_diffuse_scalar_and_shift(
            epg_states,
            t1_ms,
            t2_ms,
            repetition_time_ms - echo_time_ms,
            diffusivity,
            equilibrium_magnetization,
            wavevector_step=sequence.scalar_readout_wavevector,
            order_step=order_step,
            gradient_enabled=1,
            truncate=1,
        )
        return (epg_states, (real_part, imag_part))

    inputs = (
        jnp.array(flip_angle_train, dtype=jnp.float32),
        jnp.array(phase_train, dtype=jnp.float32),
    )
    _, (transversal_data_real, transversal_data_imag) = jax.lax.scan(
        scan_fn, epg_states, inputs
    )
    return jnp.array([transversal_data_real, transversal_data_imag])


@partial(jax.jit, static_argnums=(8, 9, 10, 11, 12, 13, 14))
def simulate_tensor_signal(
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
    """Simulate independent direction blocks, returning real/imaginary rows.

    Trains are radians, T1/T2 are ms, D is (3,3) in mm^2/s. Without inversion,
    output shape is (2,direction_count*N). With the default reused train and
    inversion it is (2,direction_count*N+1): the first direction supplies
    an inversion/readout block, remaining directions start from that block's
    final state independently. Optional sampling slices the concatenated signal.
    Preparation arrays only apply if reuse_readout_train_for_inversion=False.
    Static controls and preserved diffusion defects are documented in docs/."""
    echo_time_ms, repetition_time_ms, order_step = (
        sequence.echo_time_ms,
        sequence.repetition_time_ms,
        sequence.order_step,
    )
    epg_states = initial_epg_state(
        state_count, sequence.initial_longitudinal_magnetization
    )
    gradients = jnp.array(sequence.directions)[:direction_count]
    flip_angle_dtype = flip_angle_train.dtype

    def scan_fun(carry, inputs):
        epg_states, g = carry
        flip_angle, rf_phase = inputs
        epg_states = apply_rf_rotation(
            epg_states, flip_angle=flip_angle, rf_phase=rf_phase
        )
        epg_states = relax_diffuse_tensor_and_shift(
            epg_states,
            t1_ms,
            t2_ms,
            echo_time_ms,
            diffusion_tensor=diffusion_tensor,
            equilibrium_magnetization=equilibrium_magnetization,
            gradient_direction=g,
            wavevector_step=sequence.tensor_readout_wavevector,
            order_step=order_step,
            gradient_enabled=0,
            truncate=1,
        )
        transversal_real = jnp.real(epg_states[0, 0])
        transversal_imag = jnp.imag(epg_states[0, 0])
        epg_states = relax_diffuse_tensor_and_shift(
            epg_states,
            t1_ms,
            t2_ms,
            repetition_time_ms - echo_time_ms,
            diffusion_tensor=diffusion_tensor,
            equilibrium_magnetization=equilibrium_magnetization,
            gradient_direction=g,
            wavevector_step=sequence.tensor_readout_wavevector,
            order_step=order_step,
            gradient_enabled=1,
            truncate=1,
        )
        return ((epg_states, g), (transversal_real, transversal_imag))

    reuse_readout_train = sequence.reuse_readout_train_for_inversion
    if reuse_readout_train:
        inversion_train_angles = flip_angle_train
        inversion_train_phases = phase_train
    else:
        inversion_train_angles = preparation_flip_angles
        inversion_train_phases = preparation_phases
    if include_inversion:
        inversion_angles = jnp.array([sequence.inversion_angle])
        inversion_phase = jnp.array([sequence.inversion_rf_phase])
        full_inversion_angles = jnp.array(
            jnp.concatenate([inversion_angles, inversion_train_angles]),
            dtype=jnp.complex128,
        )
        full_inversion_phases = jnp.array(
            jnp.concatenate([inversion_phase, inversion_train_phases]),
            dtype=jnp.complex128,
        )
        prep_inputs = (full_inversion_angles, full_inversion_phases)
        g_prep = gradients[0]
        gradients = gradients[1:]
        (epg_states, _), (prep_real, prep_imag) = lax.scan(
            scan_fun, (epg_states, g_prep), prep_inputs
        )
        prep = jnp.stack([prep_real, prep_imag], axis=0)

    def process_gradient(g):
        local_states = epg_states
        local_states = apply_rf_rotation(
            local_states,
            flip_angle=sequence.preparation_angles[0],
            rf_phase=sequence.preparation_rf_phase,
        )
        local_states = relax_diffuse_tensor_and_shift(
            local_states,
            t1_ms,
            t2_ms,
            echo_time_ms,
            diffusion_tensor=diffusion_tensor,
            equilibrium_magnetization=equilibrium_magnetization,
            gradient_direction=g,
            wavevector_step=sequence.tensor_preparation_wavevector,
            order_step=1,
            gradient_enabled=1,
            truncate=1,
        )
        local_states = apply_rf_rotation(
            local_states,
            flip_angle=sequence.preparation_angles[1],
            rf_phase=sequence.preparation_rf_phase,
        )
        local_states = relax_diffuse_tensor_and_shift(
            local_states,
            t1_ms,
            t2_ms,
            echo_time_ms,
            diffusion_tensor=diffusion_tensor,
            equilibrium_magnetization=equilibrium_magnetization,
            gradient_direction=g,
            wavevector_step=sequence.tensor_preparation_wavevector,
            order_step=1,
            gradient_enabled=1,
            truncate=1,
        )
        local_states = apply_rf_rotation(
            local_states,
            flip_angle=sequence.preparation_angles[2],
            rf_phase=sequence.preparation_rf_phase,
        )
        inputs = (flip_angle_train, phase_train)
        (final_states, _), (real_main, imag_main) = lax.scan(
            scan_fun, (local_states, g), inputs
        )
        return jnp.array([real_main, imag_main])

    results = lax.map(process_gradient, gradients)
    results = results.transpose(1, 0, 2).reshape(2, -1)
    if include_inversion:
        results_processed = jnp.concatenate([prep, results], axis=1)
    else:
        results_processed = results
    if sampling:
        processed_signal = results_processed[:, sampling_offset::sampling_rate]
    else:
        processed_signal = results_processed
    return processed_signal


@partial(jax.jit, static_argnums=(6,))
def scalar_signal_magnitude(
    flip_angle_train,
    phase_train,
    t1_ms,
    t2_ms,
    diffusivity,
    equilibrium_magnetization,
    sequence=LEGACY_SEQUENCE,
):
    """Return length-N sqrt(real_signal^2+imag_signal^2); arguments match simulate_scalar_signal."""
    raw_output = simulate_scalar_signal(
        flip_angle_train,
        phase_train,
        t1_ms,
        t2_ms,
        diffusivity,
        equilibrium_magnetization,
        sequence=sequence,
    )
    return jnp.sqrt(raw_output[0] ** 2 + raw_output[1] ** 2)


@partial(jax.jit, static_argnums=(8, 9, 10, 11, 12, 13, 14))
def tensor_signal_magnitude(
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
    """Return magnitude of simulate_tensor_signal, with its direction-block/sample ordering."""
    raw_output = simulate_tensor_signal(
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
    signal = jnp.sqrt(raw_output[0] ** 2 + raw_output[1] ** 2)
    return signal
