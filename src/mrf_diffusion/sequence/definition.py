"""Explicit sequence choices; defaults reproduce the archived MSc implementation.

Wavevector units have not been calibrated to physical gradient waveforms. Do not
interpret the values below as mT/m. See docs/units.md and docs/validation.md.
"""

from dataclasses import dataclass, field
from math import pi
from typing import Any


@dataclass(frozen=True)
class SequenceSettings:
    echo_time_ms: float = 4
    repetition_time_ms: float = 15
    order_step: int = 1
    scalar_preparation_wavevector: float = 612
    scalar_readout_wavevector: float = 50
    tensor_preparation_wavevector: float = 300
    tensor_readout_wavevector: float = 100
    preparation_angles: tuple = (pi / 2, pi, -pi / 2)
    preparation_rf_phase: float = 0
    inversion_angle: float = pi
    inversion_rf_phase: float = 0
    reuse_readout_train_for_inversion: bool = True
    directions: tuple = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
    )
    # This initial value intentionally does not depend on equilibrium M.
    initial_longitudinal_magnetization: float = 1.0
    scalar_state_count: int = 150


LEGACY_SEQUENCE = SequenceSettings()


@dataclass(frozen=True)
class SimulationOptions:
    direction_count: int = 3
    state_count: int = 20
    include_inversion: bool = False
    sampling: bool = False
    sampling_offset: int = 0
    sampling_rate: int = 32


@dataclass(frozen=True)
class MRFSequence:
    """Train arrays in radians; settings are static under JIT in the kernels."""

    flip_angles: Any
    rf_phases: Any
    settings: SequenceSettings = field(default_factory=SequenceSettings)
    preparation_flip_angles: Any = None
    preparation_phases: Any = None
