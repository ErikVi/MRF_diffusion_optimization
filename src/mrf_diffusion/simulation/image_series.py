"""Complex voxel signals from unique tissue/tensor combinations; no spatial encoding."""

from dataclasses import dataclass
import numpy as np
from .api import simulate_mrf_signal
from .tissue import TissueParameters


@dataclass(frozen=True)
class ImageSeries:
    """images=(frames,H,W) complex128; frame identities follow the simulator.

    direction_index and readout_index are (frames,). block is 'readout' or
    'inversion'; inversion block starts at readout_index=-1 (the inversion RF).
    source_sample_index refers to the unsliced concatenated simulator output.
    No global acquisition time is fabricated for independently initialized blocks.
    """

    images: np.ndarray
    direction_index: np.ndarray
    readout_index: np.ndarray
    block: np.ndarray
    source_sample_index: np.ndarray
    unique_signal_count: int


def signal_frame_table(sequence, options):
    """Mirror concatenation/slicing only, not EPG evolution."""
    angles, phases = np.asarray(sequence.flip_angles), np.asarray(sequence.rf_phases)
    if (
        angles.ndim != 1
        or angles.size == 0
        or phases.shape != angles.shape
        or not (np.all(np.isfinite(angles)) and np.all(np.isfinite(phases)))
    ):
        raise ValueError("Sequence needs equal finite nonempty angle/RF phase trains")
    if not 1 <= options.direction_count <= len(sequence.settings.directions):
        raise ValueError("direction_count exceeds sequence directions")
    if (
        options.state_count < 1
        or options.sampling_rate < 1
        or options.sampling_offset < 0
    ):
        raise ValueError("Invalid simulation state count or sample slicing")
    direction, readout, block = [], [], []
    first = 0
    if options.include_inversion:
        count = len(angles)
        if not sequence.settings.reuse_readout_train_for_inversion:
            prep = np.asarray(sequence.preparation_flip_angles)
            prep_phase = np.asarray(sequence.preparation_phases)
            if (
                prep.ndim != 1
                or not len(prep)
                or prep.shape != prep_phase.shape
                or not (np.all(np.isfinite(prep)) and np.all(np.isfinite(prep_phase)))
            ):
                raise ValueError(
                    "Separate inversion block requires explicit paired preparation arrays"
                )
            count = len(prep)
        direction.extend([0] * (count + 1))
        readout.extend(range(-1, count))
        block.extend(["inversion"] * (count + 1))
        first = 1
    for d in range(first, options.direction_count):
        direction.extend([d] * len(angles))
        readout.extend(range(len(angles)))
        block.extend(["readout"] * len(angles))
    indices = np.arange(len(direction))
    if options.sampling:
        indices = indices[options.sampling_offset :: options.sampling_rate]
    if len(indices) == 0:
        raise ValueError("Simulation slicing selects no samples")
    return (
        np.asarray(direction)[indices],
        np.asarray(readout)[indices],
        np.asarray(block)[indices],
        indices,
    )


def simulate_phantom_image_series(phantom, sequence, options):
    """Simulate each exact unique (T1,T2,D) once via the authoritative tensor API.

    M=1 is fixed. Output = proton_density * exp(i*object_phase_map) * complex
    simulator signal. RF phase is already applied inside EPG and is NEVER added
    here. Exact uniqueness avoids merging physically distinct nearby tissues.
    Background is not simulated. This host adapter retains existing kernel defects.
    """
    table = signal_frame_table(sequence, options)
    active = phantom.support
    parameters = np.column_stack(
        (
            phantom.t1_ms[active],
            phantom.t2_ms[active],
            phantom.diffusion_tensor[active].reshape(-1, 9),
        )
    )
    unique, inverse = np.unique(parameters, axis=0, return_inverse=True)
    signals = []
    for row in unique:
        raw = np.asarray(
            simulate_mrf_signal(
                TissueParameters(
                    t1_ms=float(row[0]),
                    t2_ms=float(row[1]),
                    equilibrium_magnetization=1.0,
                    diffusion=row[2:].reshape(3, 3),
                ),
                sequence,
                options,
                tensor=True,
            )
        )
        if raw.shape != (2, len(table[0])) or not np.all(np.isfinite(raw)):
            raise ValueError(
                "Simulator output disagrees with frame table or is nonfinite"
            )
        signals.append(raw[0] + 1j * raw[1])
    images = np.zeros((len(table[0]), *active.shape), complex)
    scale = phantom.proton_density[active] * np.exp(
        1j * phantom.object_phase_map[active]
    )
    images[:, active] = np.asarray(signals)[inverse].T * scale[None, :]
    return ImageSeries(images, *table, len(unique))
