"""Experiment defaults reproduce the checked-in scripts, not all thesis figures."""

from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pathlib import Path
import argparse
import json
import tomllib
from mrf_diffusion.sequence.definition import SequenceSettings, SimulationOptions
from mrf_diffusion.optimization.settings import ConstraintSettings, SolverSettings


@dataclass(frozen=True)
class TissueEnsemble:
    scalar_parameters: tuple = (
        (1000.0, 70.0, 0.7),
        (1000.0, 70.0, 0.7),
        (1500.0, 100.0, 0.8),
    )
    diffusion_tensors: tuple = (
        ((0.00089, 0.0001, 0.0001), (0.0001, 0.0002, 0.0001), (0.0001, 0.0001, 0.0002)),
        ((0.0007, 0.0001, 0.0001), (0.0001, 0.0003, 0.0001), (0.0001, 0.0001, 0.0003)),
        ((0.00075, 0.0001, 0.0001), (0.0001, 0.0004, 0.0001), (0.0001, 0.0001, 0.0004)),
    )
    parameter_weights: tuple = (1, 1, 1, 1, 1)
    tissue_weights: tuple = (1 / 3, 1 / 3, 1 / 3)


@dataclass(frozen=True)
class ExperimentSettings:
    input_directory: str = "data/input"
    output_directory: str = "data/output"
    initial_angle_file: str = "fa_array_initial.npy"
    preparation_angle_file: str = "DIFFPREPARATION/fa_train_diffprep.npy"
    preparation_phase_file: str = "DIFFPREPARATION/pm_train_diffprep.npy"
    length: int = 400
    knot_setting: int = 50
    degree: int = 3
    phase_methods: tuple = (
        "free form",
        "no phase modulation",
        "quadratic",
        "quadratic malleable",
    )
    aggregation: str = "L1"
    initial_phase_fraction: float = 0.18
    initial_phase_method: str = "quadratic"
    piecewise_parameters: tuple = (0.3, 0.04, -0.03)
    phase_scan_count: int = 400
    phase_scan_minimum: float = 0.01
    phase_scan_maximum: float = 1.0
    benchmark_repetitions: int = 10000
    plot_diffusion_minimum: float = 5e-4
    plot_diffusion_maximum: float = 1e-3
    plot_diffusion_samples: int = 1000
    plot_tissue: tuple = (1500.0, 90.0, 0.9)
    plot_tensor_template: tuple = ((1.8, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.1, 0.2))
    sequence: SequenceSettings = field(default_factory=SequenceSettings)
    simulation: SimulationOptions = field(
        default_factory=lambda: SimulationOptions(
            direction_count=1, state_count=5, include_inversion=True
        )
    )
    tissues: TissueEnsemble = field(default_factory=TissueEnsemble)
    constraints: ConstraintSettings = field(default_factory=ConstraintSettings)
    solver: SolverSettings = field(default_factory=SolverSettings)


def _tuple_values(value):
    return tuple(_tuple_values(x) for x in value) if isinstance(value, list) else value


def _update(instance, data):
    known = {f.name for f in fields(instance)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {type(instance).__name__} fields: {sorted(unknown)}")
    values = {f.name: getattr(instance, f.name) for f in fields(instance)}
    for key, value in data.items():
        values[key] = (
            _update(values[key], value)
            if is_dataclass(values[key])
            else _tuple_values(value)
        )
    return type(instance)(**values)


def parse_settings(description, defaults, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML configuration; relative paths resolve beside the file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without loading data or running computations",
    )
    args = parser.parse_args(argv)
    settings = defaults
    if args.config:
        config_path = args.config.resolve()
        settings = _update(
            defaults, tomllib.loads(config_path.read_text(encoding="utf-8"))
        )
        settings = _update(
            settings,
            {
                name: str((config_path.parent / getattr(settings, name)).resolve())
                for name in ("input_directory", "output_directory")
            },
        )
    if args.dry_run:
        print(json.dumps(asdict(settings), indent=2))
        return None
    return settings


def required_inputs(settings, preparation=True):
    root = Path(settings.input_directory)
    names = [settings.initial_angle_file]
    if preparation:
        names += [settings.preparation_angle_file, settings.preparation_phase_file]
    paths = [root / name for name in names]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing experiment inputs (no substitute data generated): "
            + ", ".join(missing)
        )
    return paths
