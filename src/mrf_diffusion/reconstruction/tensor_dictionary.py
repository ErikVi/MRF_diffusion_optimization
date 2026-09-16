"""Bounded complex tensor dictionaries from the authoritative MRF simulator.

Grid axes are independent; MD/FA select a valid prolate tensor, not independent
signal corrections. M=1; density and constant object phase are fitted analytically.
"""

from dataclasses import dataclass
from itertools import product
from math import prod
import numpy as np
from mrf_diffusion.diffusion.parameterization import axisymmetric_tensor
from mrf_diffusion.simulation.api import simulate_mrf_signal
from mrf_diffusion.simulation.tissue import TissueParameters
from mrf_diffusion.simulation.image_series import signal_frame_table


@dataclass(frozen=True)
class DictionaryGrid:
    t1_ms: tuple = (750.0, 1250.0)
    t2_ms: tuple = (70.0, 90.0)
    mean_diffusivity_mm2_per_s: tuple = (0.0007, 0.001)
    fractional_anisotropy: tuple = (0.2, 0.7)
    principal_directions_xyz: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    max_entries: int = 10000
    max_working_bytes: int = 268435456
    voxel_batch_size: int = 256


@dataclass(frozen=True)
class TensorDictionary:
    """signals (entries,frames), t1/t2 (entries), tensors (entries,3,3)."""

    signals: np.ndarray
    t1_ms: np.ndarray
    t2_ms: np.ndarray
    tensors: np.ndarray


def estimate_dictionary_size(grid, frame_count):
    """Conservative host-array estimate; excludes JAX compilation/device memory."""
    axes = (
        grid.t1_ms,
        grid.t2_ms,
        grid.mean_diffusivity_mm2_per_s,
        grid.fractional_anisotropy,
        grid.principal_directions_xyz,
    )
    if any(len(axis) == 0 for axis in axes):
        raise ValueError("Dictionary axes must be nonempty")
    if any(
        not isinstance(v, int) or isinstance(v, bool) or v <= 0
        for v in (
            frame_count,
            grid.max_entries,
            grid.max_working_bytes,
            grid.voxel_batch_size,
        )
    ):
        raise ValueError("Dictionary dimensions and budgets must be positive integers")
    entries = prod(map(len, axes))
    signal_bytes = entries * frame_count * 16
    # Signals, normalized/conjugated copies, parameter rows and matching temporaries.
    working = signal_bytes * 4 + entries * (256 + grid.voxel_batch_size * 48)
    return dict(
        axis_counts=list(map(len, axes)),
        entries_upper_bound=entries,
        frames=frame_count,
        signal_bytes=signal_bytes,
        estimated_host_working_bytes=working,
        amplitude_samples=0,
        amplitude_handling="analytic complex least squares",
    )


def generate_tensor_dictionary(grid, sequence, options):
    frames = len(signal_frame_table(sequence, options)[0])
    estimate = estimate_dictionary_size(grid, frames)
    if (
        estimate["entries_upper_bound"] > grid.max_entries
        or estimate["estimated_host_working_bytes"] > grid.max_working_bytes
    ):
        raise ValueError(
            f"Dictionary budget exceeded before simulation: {estimate}. Restrict the grid or design coarse-to-fine fitting."
        )
    for axis in (grid.t1_ms, grid.t2_ms):
        values = np.asarray(axis)
        if (
            np.iscomplexobj(values)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError("T1/T2 grid values must be finite and positive")
    rows = []
    for t1, t2, md, fa, direction in product(
        grid.t1_ms,
        grid.t2_ms,
        grid.mean_diffusivity_mm2_per_s,
        grid.fractional_anisotropy,
        grid.principal_directions_xyz,
    ):
        tensor = axisymmetric_tensor(md, fa, direction)
        rows.append(np.r_[t1, t2, tensor.ravel()])
    # Removes exact physical duplicates, including antipodal/isotropic orientations.
    rows = np.unique(np.asarray(rows), axis=0)
    signals = []
    for row in rows:
        channels = np.asarray(
            simulate_mrf_signal(
                TissueParameters(
                    float(row[0]), float(row[1]), 1.0, row[2:].reshape(3, 3)
                ),
                sequence,
                options,
                tensor=True,
            )
        )
        if channels.shape != (2, frames) or not np.all(np.isfinite(channels)):
            raise ValueError("Invalid dictionary signal from simulator")
        signals.append(channels[0] + 1j * channels[1])
    signals = np.asarray(signals)
    if np.any(np.linalg.norm(signals, axis=1) == 0):
        raise ValueError("Zero dictionary atoms cannot be matched")
    return TensorDictionary(
        signals, rows[:, 0], rows[:, 1], rows[:, 2:].reshape(-1, 3, 3)
    )
