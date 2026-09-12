"""Benchmark the canonical spline evaluator; old duplicate algorithms were retired."""

from dataclasses import replace
from time import perf_counter
import tracemalloc
import numpy as np
import jax
import jax.numpy as jnp
from .settings import ExperimentSettings, parse_settings, required_inputs
from mrf_diffusion.sequence.bspline import evaluate_bspline, fit_bspline_coefficients
from mrf_diffusion.sequence.parameterization import make_legacy_knots
from mrf_diffusion.io.results import save_run

DEFAULTS = replace(ExperimentSettings(), length=300, knot_setting=20)


def run(settings):
    angles = jnp.asarray(np.load(required_inputs(settings, preparation=False)[0]))[
        : settings.length
    ]
    knots = make_legacy_knots(len(angles), settings.knot_setting, settings.degree)
    points = jnp.linspace(0, len(angles) - 1, len(angles))
    coefficients = fit_bspline_coefficients(angles, knots, settings.degree)
    jax.block_until_ready(
        evaluate_bspline(points, knots, coefficients, settings.degree)
    )
    start = perf_counter()
    for _ in range(settings.benchmark_repetitions):
        jax.block_until_ready(
            evaluate_bspline(points, knots, coefficients, settings.degree)
        )
    elapsed = perf_counter() - start
    tracemalloc.start()
    jax.block_until_ready(
        evaluate_bspline(points, knots, coefficients, settings.degree)
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "seconds_per_call": elapsed / settings.benchmark_repetitions,
        "python_peak_bytes": peak,
    }
    save_run(settings.output_directory, settings, {}, result)
    print(result)
    return result


def main(argv=None):
    settings = parse_settings(__doc__, DEFAULTS, argv)
    if settings is not None:
        run(settings)


if __name__ == "__main__":
    main()
