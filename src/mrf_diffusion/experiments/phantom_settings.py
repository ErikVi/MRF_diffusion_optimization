"""Explicit settings of the historical undersampling experiment."""

from dataclasses import dataclass, field
from mrf_diffusion.sequence.definition import SequenceSettings, SimulationOptions


@dataclass(frozen=True)
class PhantomSettings:
    input_directory: str = "data/input"
    output_directory: str = "data/output/phantom"
    initial_angle_file: str = "fa_array_initial.npy"
    preparation_angle_file: str = "DIFFPREPARATION/fa_train_diffprep.npy"
    preparation_phase_file: str = "DIFFPREPARATION/pm_train_diffprep.npy"
    length: int = 170
    phase_parameters: tuple = (0.5, 0.03, -0.03)
    grid_count: int = 30
    t1_minimum: float = 150
    t1_maximum: float = 3000
    t2_minimum: float = 30
    t2_maximum: float = 1000
    scale_minimum: float = 0
    scale_maximum: float = 1
    tile_count: int = 11
    tile_size: int = 11
    mask_fraction: float = 0.7
    white_t1_ms: float = 750
    gray_t1_ms: float = 1250
    white_t2_ms: float = 70
    gray_t2_ms: float = 90
    base_tensor: tuple = (
        (0.001, 0.0001, 0.0001),
        (0.0001, 0.001, 0.0001),
        (0.0001, 0.0001, 0.001),
    )
    add_noise: bool = False
    snr_label: float = 200
    noise_standard_deviation: float = 0.005
    random_seed: int | None = None
    density_compensation: bool = True
    golden_angle: bool = False
    spiral_name: str = "Philips_spiral"
    offset: int = 0
    phase_field_order: int = 2
    sequence: SequenceSettings = field(default_factory=SequenceSettings)
    simulation: SimulationOptions = field(
        default_factory=lambda: SimulationOptions(
            direction_count=9, state_count=5, include_inversion=False
        )
    )
