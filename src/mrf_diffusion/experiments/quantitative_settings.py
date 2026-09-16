"""Bounded quantitative validation and paired sequence comparison settings."""

from dataclasses import dataclass, field
from mrf_diffusion.experiments.forward_settings import ForwardSettings
from mrf_diffusion.reconstruction.tensor_dictionary import DictionaryGrid


@dataclass(frozen=True)
class ComparisonSettings:
    # Paths are complete forward TOMLs with actual paired sequence archives.
    flip_optimized_config: str = ""
    joint_optimized_config: str = ""
    interleaf_counts: tuple = (1, 4, 16)
    noise_std_per_channel: tuple = (0.0, 0.001)
    noise_seeds: tuple = (2026,)
    density_compensation: str = "radial_increment"
    unit_impulse_gain: bool = True
    # Gates are checked before ANY undersampled condition.
    reference_image_relative_tolerance: float = 1e-4
    reference_density_relative_tolerance: float = 1e-4
    ambiguity_tolerance: float = 1e-10


@dataclass(frozen=True)
class QuantitativeSettings(ForwardSettings):
    output_directory: str = "data/output/quantitative_undersampling"
    dictionary: DictionaryGrid = field(default_factory=DictionaryGrid)
    comparison: ComparisonSettings = field(default_factory=ComparisonSettings)
