"""Compare phase families using the current three-bound objective.

The malformed historical no-phase coefficient baseline is retained explicitly;
this experiment must not be presented as corrected five-parameter optimization.
"""

from dataclasses import replace, asdict
from functools import partial
import jax.numpy as jnp
import numpy as np
from .settings import ExperimentSettings, parse_settings, required_inputs
from mrf_diffusion.sequence.definition import SimulationOptions
from mrf_diffusion.sequence.parameterization import make_legacy_knots
from mrf_diffusion.sequence.bspline import fit_bspline_coefficients
from mrf_diffusion.sequence.phase import generate_phase_train
from mrf_diffusion.optimization.evaluation import tensor_sequence_precision_bounds
from mrf_diffusion.io.results import save_run

DEFAULTS = replace(
    ExperimentSettings(),
    length=170,
    phase_methods=("quadratic", "linear", "sinusoidal", "alternating"),
    simulation=SimulationOptions(direction_count=9, state_count=5),
)


def run(settings):
    if settings.degree != 3:
        raise ValueError(
            "The historical phase-comparison objective supports cubic splines only"
        )
    angles = jnp.asarray(np.load(required_inputs(settings, preparation=False)[0]))[
        : settings.length
    ]
    points = jnp.linspace(0, len(angles) - 1, len(angles))
    knots = make_legacy_knots(len(angles), settings.knot_setting, settings.degree)
    angle_coefficients = fit_bspline_coefficients(angles, knots, settings.degree)
    evaluate = partial(
        tensor_sequence_precision_bounds,
        eval_points=points,
        knots=knots,
        scalar_tissues=jnp.asarray(settings.tissues.scalar_parameters),
        diffusion_tensors=jnp.asarray(settings.tissues.diffusion_tensors),
        parameter_weights=jnp.asarray(settings.tissues.parameter_weights),
        tissue_weights=jnp.asarray(settings.tissues.tissue_weights),
        phase_offset=0.0,
        phase_slope=0.0,
        preparation_flip_angles=jnp.zeros(3),
        preparation_phases=jnp.zeros(3),
        **asdict(settings.simulation),
        sequence=settings.sequence,
        method="free form",
    )
    baseline = evaluate(jnp.concatenate((angle_coefficients, jnp.zeros(len(angles)))))
    fractions = jnp.linspace(
        settings.phase_scan_minimum,
        settings.phase_scan_maximum,
        settings.phase_scan_count,
    )
    results = {}
    for method in settings.phase_methods:
        values = []
        for fraction in fractions:
            phases = generate_phase_train(angles, fraction, method)
            coefficients = fit_bspline_coefficients(phases, knots, settings.degree)
            values.append(evaluate(jnp.concatenate((angle_coefficients, coefficients))))
        results[method] = np.asarray(values)
    best_method = min(results, key=lambda name: results[name].sum(axis=1).min())
    best_index = int(np.argmin(results[best_method].sum(axis=1)))
    save_run(
        settings.output_directory,
        settings,
        {"fractions": fractions, "legacy_baseline": baseline, **results},
        {
            "best_method": best_method,
            "best_fraction": float(fractions[best_index]),
            "parameter_labels": ["T1", "T2", "equilibrium M"],
            "warning": "Legacy malformed baseline and three-parameter objective retained",
        },
    )
    from mrf_diffusion.visualization.plots import plot_phase_comparison

    plot_phase_comparison(fractions, results, baseline, settings.output_directory)
    return results


def main(argv=None):
    settings = parse_settings(__doc__, DEFAULTS, argv)
    if settings is not None:
        run(settings)


if __name__ == "__main__":
    main()
