"""MSc sequence optimization, configured per run and independent of plotting."""

from dataclasses import asdict
from functools import partial
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from .settings import ExperimentSettings, parse_settings, required_inputs
from mrf_diffusion.sequence.parameterization import (
    make_legacy_knots,
    initialize_sequence_parameters,
    decode_sequence_parameters,
)
from mrf_diffusion.optimization.constraints import make_sequence_constraints
from mrf_diffusion.optimization.objectives import tensor_ensemble_objective
from mrf_diffusion.optimization.evaluation import tensor_sequence_precision_bounds
from mrf_diffusion.optimization.solver import optimize_sequence
from mrf_diffusion.simulation.signal import tensor_signal_magnitude
from mrf_diffusion.io.results import save_run


def build_problem(settings, flip_angles, prep_angles, prep_phases, method):
    if settings.degree != 3:
        raise ValueError(
            "The historical optimization objective supports cubic splines only"
        )
    points = jnp.linspace(0, len(flip_angles) - 1, len(flip_angles))
    knots = make_legacy_knots(len(flip_angles), settings.knot_setting, settings.degree)
    initial, phase, offset, slope = initialize_sequence_parameters(
        flip_angles,
        points,
        knots,
        method,
        settings.degree,
        settings.initial_phase_fraction,
        settings.initial_phase_method,
        settings.piecewise_parameters,
    )
    arguments = dict(
        eval_points=points,
        knots=knots,
        scalar_tissues=jnp.asarray(settings.tissues.scalar_parameters),
        diffusion_tensors=jnp.asarray(settings.tissues.diffusion_tensors),
        parameter_weights=jnp.asarray(settings.tissues.parameter_weights),
        tissue_weights=jnp.asarray(settings.tissues.tissue_weights),
        phase_offset=offset,
        phase_slope=slope,
        preparation_flip_angles=prep_angles,
        preparation_phases=prep_phases,
        **asdict(settings.simulation),
        method=method,
        sequence=settings.sequence,
    )
    objective = partial(
        tensor_ensemble_objective, **arguments, aggregation=settings.aggregation
    )
    bounds = partial(tensor_sequence_precision_bounds, **arguments)
    constraints = make_sequence_constraints(points, knots, method, settings.constraints)
    return initial, phase, points, knots, offset, slope, objective, bounds, constraints


def run(settings):
    angle_path, prep_angle_path, prep_phase_path = required_inputs(settings)
    angles = jnp.asarray(np.load(angle_path))[: settings.length]
    prep_angles, prep_phases = jnp.asarray(np.load(prep_angle_path)), jnp.asarray(
        np.load(prep_phase_path)
    )
    from mrf_diffusion.visualization.plots import (
        plot_train,
        plot_precision,
        plot_signal_family,
    )

    summaries = []
    for method in settings.phase_methods:
        (
            initial,
            initial_phase,
            points,
            knots,
            offset,
            slope,
            objective,
            bounds,
            constraints,
        ) = build_problem(settings, angles, prep_angles, prep_phases, method)
        result, history = optimize_sequence(
            objective,
            initial,
            constraints,
            settings.solver,
            settings.constraints,
            bounds,
        )
        optimized_angles, optimized_phases = decode_sequence_parameters(
            result.x, points, knots, offset, slope, method
        )
        initial_bounds, optimized_bounds = bounds(initial), bounds(result.x)
        directory = Path(settings.output_directory) / method.replace(" ", "_")
        summary = {
            "method": method,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "objective": float(result.fun),
            "iterations": int(result.nit),
            "history": history,
            "initial_bounds": initial_bounds.tolist(),
            "optimized_bounds": optimized_bounds.tolist(),
        }
        save_run(
            directory,
            settings,
            {
                "optimized_fa_train": optimized_angles,
                "optimized_phase_modulation": optimized_phases,
                "initial_coefficients": initial,
                "optimized_coefficients": result.x,
            },
            summary,
        )
        plot_train(optimized_angles, directory, "flip_angles.png", "Flip angle [rad]")
        plot_train(optimized_phases, directory, "rf_phases.png", "RF phase [rad]")
        plot_train(angles, directory, "initial_flip_angles.png", "Flip angle [rad]")
        plot_train(initial_phase, directory, "initial_rf_phases.png", "RF phase [rad]")
        plot_precision(initial_bounds, optimized_bounds, directory)
        scales = jnp.logspace(
            np.log10(settings.plot_diffusion_minimum),
            np.log10(settings.plot_diffusion_maximum),
            settings.plot_diffusion_samples,
        )
        signals = [
            tensor_signal_magnitude(
                optimized_angles,
                optimized_phases,
                *settings.plot_tissue,
                jnp.asarray(settings.plot_tensor_template) * scale,
                prep_angles,
                prep_phases,
                **asdict(settings.simulation),
                sequence=settings.sequence,
            )
            for scale in scales
        ]
        plot_signal_family(signals, scales, directory)
        summaries.append(summary)
        print(
            f"{method}: success={result.success}; {result.message}; objective={result.fun}"
        )
    return summaries


def main(argv=None):
    settings = parse_settings(__doc__, ExperimentSettings(), argv)
    if settings is not None:
        run(settings)


if __name__ == "__main__":
    main()
