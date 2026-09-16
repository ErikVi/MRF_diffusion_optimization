"""Explicit configuration for the forward-only phantom experiment."""

from dataclasses import dataclass, field
from mrf_diffusion.sequence.definition import SequenceSettings, SimulationOptions


@dataclass(frozen=True)
class PhantomGeometrySettings:
    image_shape: tuple = (16, 16)
    tile_shape: tuple = (4, 4)
    geometry: str = "checkerboard"
    radius_pixels: float | None = None
    t1_ms: tuple = (750.0, 1250.0)
    t2_ms: tuple = (70.0, 90.0)
    proton_density: tuple = (1.0, 0.9)


@dataclass(frozen=True)
class DiffusionMapSettings:
    mean_diffusivity_mm2_per_s: tuple = (0.0007, 0.001)
    fractional_anisotropy: tuple = (0.2, 0.7)
    principal_directions_xyz: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


@dataclass(frozen=True)
class TrainSettings:
    flip_angles_rad: tuple = (0.2, 0.3, 0.4, 0.25, 0.5, 0.35, 0.2, 0.45)
    rf_phase_mode: str = "reference"
    rf_phases_rad: tuple = ()
    reference_method: str = "quadratic"
    reference_fraction: float = 0.18
    optimized_archive: str = ""
    physics: SequenceSettings = field(default_factory=SequenceSettings)


@dataclass(frozen=True)
class TrajectorySettings:
    kind: str = "generated_spiral"
    supplied_archive: str = ""
    layout: str = "components_first"
    component_order: str = "xy"
    schedule: str = "golden"
    interleaves_per_frame: int = 1
    angular_increment_rad: float = 0.19634954084936207
    initial_angle_rad: float = 0.0
    reference_arm_count: int = 32
    field_of_view_m: float = 0.24
    design_matrix_size: int = 16
    frequency_sampling_factor: float = 1.0
    acceleration: float = 4.0
    density_exponent: float = 6.0
    max_gradient_t_per_m: float = 0.03
    max_slew_t_per_m_per_s: float = 150.0
    oversampling: float = 2.0
    kernel_width: float = 6.0


@dataclass(frozen=True)
class ObjectPhaseSettings:
    enabled: bool = False
    edge_phase_rad: float = 0.0


@dataclass(frozen=True)
class NoiseSettings:
    enabled: bool = False
    standard_deviation_per_channel: float = 0.001
    seed: int = 2026


@dataclass(frozen=True)
class DensitySettings:
    method: str = "none"
    pipe_menon_iterations: int = 30


@dataclass(frozen=True)
class ReconstructionSettings:
    enabled: bool = True
    representative_frames: tuple = (0, 3, 7)


@dataclass(frozen=True)
class ForwardSettings:
    input_directory: str = "data/input"
    output_directory: str = "data/output/forward_undersampling"
    phantom: PhantomGeometrySettings = field(default_factory=PhantomGeometrySettings)
    diffusion: DiffusionMapSettings = field(default_factory=DiffusionMapSettings)
    sequence: TrainSettings = field(default_factory=TrainSettings)
    simulation: SimulationOptions = field(
        default_factory=lambda: SimulationOptions(direction_count=2, state_count=20)
    )
    trajectory: TrajectorySettings = field(default_factory=TrajectorySettings)
    object_phase: ObjectPhaseSettings = field(default_factory=ObjectPhaseSettings)
    noise: NoiseSettings = field(default_factory=NoiseSettings)
    density_compensation: DensitySettings = field(default_factory=DensitySettings)
    reconstruction: ReconstructionSettings = field(
        default_factory=ReconstructionSettings
    )
