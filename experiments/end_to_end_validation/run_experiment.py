"""Reproducible orchestration; MRI and reconstruction remain package operations."""

from pathlib import Path
from dataclasses import asdict, replace
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import tomllib
import zipfile
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mrf_diffusion.experiments.settings import (
    ExperimentSettings,
    TissueEnsemble,
    _update,
)
from mrf_diffusion.experiments.optimization import build_problem
from mrf_diffusion.optimization.solver import optimize_sequence
from mrf_diffusion.sequence.parameterization import decode_sequence_parameters
from mrf_diffusion.sequence.definition import MRFSequence, SimulationOptions
from mrf_diffusion.diffusion.parameterization import axisymmetric_tensor
from mrf_diffusion.experiments.quantitative_settings import QuantitativeSettings
from mrf_diffusion.experiments.forward_undersampling import (
    build_phantom,
    build_trajectories,
)
from mrf_diffusion.experiments.quantitative_undersampling import (
    truth_and_regions,
    reference_recovery_gate,
)
from mrf_diffusion.simulation.image_series import simulate_phantom_image_series
from mrf_diffusion.reconstruction.tensor_dictionary import (
    generate_tensor_dictionary,
    estimate_dictionary_size,
)
from mrf_diffusion.encoding.series import acquire_image_series
from mrf_diffusion.reconstruction.calibrated import reconstruct_with_impulse_gain
from mrf_diffusion.reconstruction.quantitative import match_complex_dictionary
from mrf_diffusion.reconstruction.metrics import evaluate_parameter_maps


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def save_figure(root, name, figure):
    figure.savefig(root / "figures" / (name + ".png"), dpi=300, bbox_inches="tight")
    figure.savefig(root / "figures" / (name + ".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(config_path):
    start = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = tomllib.loads(config_path.read_text())
    root = (
        config_path.parent / config["output_directory"] / config["run_name"]
    ).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "config",
        "sequences",
        "optimization",
        "signals",
        "phantom",
        "acquisition",
        "reconstruction",
        "parameter_maps",
        "metrics",
        "figures",
        "report",
    ):
        (root / name).mkdir()
    (root / "config" / "experiment.toml").write_bytes(config_path.read_bytes())
    repo = Path(__file__).resolve().parents[2]
    metadata = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        versions={
            n: importlib.metadata.version(n)
            for n in ("jax", "jaxlib", "numpy", "scipy", "sigpy")
        },
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        git_status=subprocess.check_output(
            ["git", "status", "--short"], cwd=repo, text=True
        ),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
    )
    with zipfile.ZipFile(
        root / "config" / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for folder in ("src", "experiments/end_to_end_validation", "tests", "docs"):
            for path in (repo / folder).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(repo))
        for path in (repo / "pyproject.toml", repo / "AGENTS.md"):
            archive.write(path, path.relative_to(repo))
    write_json(root / "config" / "metadata.json", metadata)
    settings = _update(QuantitativeSettings(), config.get("imaging", {}))
    size = config["image_size"]
    options = SimulationOptions(direction_count=3, state_count=config["state_count"])
    settings = replace(
        settings,
        simulation=options,
        phantom=replace(
            settings.phantom,
            image_shape=(size, size),
            tile_shape=(size // 4, size // 4),
        ),
        trajectory=replace(
            settings.trajectory,
            design_matrix_size=size,
            acceleration=min(4.0, size / 4),
        ),
        object_phase=replace(
            settings.object_phase,
            enabled=True,
            edge_phase_rad=config["object_phase_edge_rad"],
        ),
    )
    tensors = tuple(
        tuple(map(tuple, axisymmetric_tensor(md, fa, axis)))
        for md, fa, axis in zip(
            settings.diffusion.mean_diffusivity_mm2_per_s,
            settings.diffusion.fractional_anisotropy,
            settings.diffusion.principal_directions_xyz,
        )
    )
    ensemble = TissueEnsemble(
        scalar_parameters=tuple(
            (t1, t2, 1.0)
            for t1, t2 in zip(settings.phantom.t1_ms, settings.phantom.t2_ms)
        ),
        diffusion_tensors=tensors,
        tissue_weights=(0.5, 0.5),
    )
    optimization = replace(
        _update(ExperimentSettings(), config.get("optimizer", {})),
        length=config["length"],
        knot_setting=config["knot_setting"],
        simulation=options,
        tissues=ensemble,
        initial_phase_fraction=config["initial_phase_fraction"],
    )
    optimization = replace(
        optimization,
        solver=replace(
            optimization.solver, max_iterations=config["maximum_iterations"]
        ),
    )
    write_json(
        root / "config" / "resolved.json",
        dict(
            imaging=asdict(settings),
            optimization=asdict(optimization),
            experiment=config,
        ),
    )
    target = (
        config["initial_minimum_rad"]
        + config["initial_excursion_rad"]
        * np.sin(np.linspace(0, np.pi, config["length"])) ** 2
    )
    prep = jnp.zeros(3)
    initial, _, points, knots, offset, slope, objective, precision, constraints = (
        build_problem(optimization, jnp.asarray(target), prep, prep, "quadratic")
    )
    if min(float(np.min(c["fun"](initial))) for c in constraints) < -1e-8:
        raise ValueError("Educated initialization violates existing constraints")
    baseline_objective = float(objective(initial))
    print("Baseline objective", baseline_objective, flush=True)
    optimization_start = time.perf_counter()
    result, history = optimize_sequence(
        objective, initial, constraints, optimization.solver, optimization.constraints
    )
    optimization_seconds = time.perf_counter() - optimization_start
    final_objective = float(objective(result.x))
    optimization_summary = dict(
        initial_objective=baseline_objective,
        final_objective=final_objective,
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        runtime_seconds=optimization_seconds,
        minimum_constraint=float(min(np.min(c["fun"](result.x)) for c in constraints)),
        history=history,
    )
    write_json(root / "optimization" / "summary.json", optimization_summary)
    np.savez(
        root / "optimization" / "parameters.npz",
        initial=initial,
        optimized=result.x,
        knots=knots,
        points=points,
        target_flip_angles_rad=target,
        phase_offset=offset,
        phase_slope=slope,
    )
    if (
        not np.isfinite(final_objective)
        or optimization_summary["minimum_constraint"] < -1e-6
    ):
        raise ValueError(
            "Optimizer produced invalid or infeasible sequence; inspect saved diagnostics"
        )
    sequences = {}
    bounds = {}
    for name, coefficients in (("initial", initial), ("optimized", result.x)):
        angles, phases = decode_sequence_parameters(
            coefficients, points, knots, offset, slope, "quadratic"
        )
        sequences[name] = MRFSequence(angles, phases, optimization.sequence)
        np.savez(
            root / "sequences" / (name + ".npz"),
            flip_angles_rad=angles,
            rf_phases_rad=phases,
        )
        np.save(root / "sequences" / (name + "_flip_angles.npy"), angles)
        np.save(root / "sequences" / (name + "_rf_phases.npy"), phases)
        np.savetxt(
            root / "sequences" / (name + ".csv"),
            np.column_stack((angles, phases)),
            delimiter=",",
            header="flip_angle_rad,rf_phase_rad",
            comments="",
        )
        bounds[name] = np.asarray(precision(coefficients)).tolist()
    write_json(
        root / "metrics" / "legacy_relative_sd_bounds.json",
        dict(
            parameter_order=["T1", "T2", "equilibrium_M"],
            values=bounds,
            MD=None,
            FA=None,
            reason="Established objective conditions on diffusion; legacy MD/FA derivatives are invalid.",
        ),
    )
    from information_diagnostic import evaluate

    information = {
        name: evaluate(sequence, options, root / "signals", name)
        for name, sequence in sequences.items()
    }
    write_json(root / "metrics" / "physical_parameter_information.json", information)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for name, seq in sequences.items():
        axes[0].plot(seq.flip_angles, label=name)
        axes[1].plot(seq.rf_phases, label=name)
    axes[0].set_ylabel("Flip angle (rad)")
    axes[1].set_ylabel("RF phase (rad)")
    axes[1].set_xlabel("Readout index")
    axes[0].legend()
    save_figure(root, "initial_vs_optimized_trains", fig)
    fig, ax = plt.subplots()
    ax.plot(
        [0] + [h["iteration"] for h in history],
        [baseline_objective] + [h["objective"] for h in history],
    )
    ax.set(xlabel="Iteration", ylabel="Legacy weighted relative SD objective")
    save_figure(root, "optimization_history", fig)
    fig, ax = plt.subplots()
    x = np.arange(3)
    for i, (name, values) in enumerate(bounds.items()):
        ax.bar(x + i * 0.35, values, width=0.35, label=name)
    ax.set_xticks(x + 0.175, ["T1", "T2", "equilibrium M"])
    ax.set_ylabel("Weighted relative SD bound")
    ax.legend()
    save_figure(root, "conditional_crlb_comparison", fig)
    phantom = build_phantom(settings)
    truth, regions = truth_and_regions(phantom)
    np.savez(
        root / "phantom" / "ground_truth.npz",
        **truth,
        support=phantom.support,
        regions=regions,
        diffusion_tensor=phantom.diffusion_tensor,
        object_phase_map=phantom.object_phase_map,
    )
    prepared = {}
    gates = {}
    for name, sequence in sequences.items():
        print("Simulating and checking", name, flush=True)
        series = simulate_phantom_image_series(phantom, sequence, options)
        write_json(
            root / "config" / "dictionary_budget.json",
            estimate_dictionary_size(settings.dictionary, len(series.images)),
        )
        dictionary = generate_tensor_dictionary(settings.dictionary, sequence, options)
        reference = build_trajectories(
            replace(
                settings,
                trajectory=replace(
                    settings.trajectory, kind="cartesian", interleaves_per_frame=1
                ),
            ),
            len(series.images),
        )
        gates[name] = reference_recovery_gate(
            settings, phantom, series, dictionary, reference
        )
        write_json(root / "metrics" / "reference_gates.json", gates)
        np.savez(
            root / "signals" / (name + ".npz"),
            images=series.images,
            dictionary=dictionary.signals,
            t1_ms=dictionary.t1_ms,
            t2_ms=dictionary.t2_ms,
            tensors=dictionary.tensors,
        )
        if not gates[name]["passed"]:
            raise ValueError(
                "Reference recovery failed; undersampling interpretation stopped"
            )
        prepared[name] = (series, dictionary)
    records = []
    conditions = [("cartesian", 1, reference)]
    for arms in config["interleaf_counts"]:
        coordinates = build_trajectories(
            replace(
                settings,
                trajectory=replace(settings.trajectory, interleaves_per_frame=arms),
            ),
            len(series.images),
        )
        conditions.append(("spiral", arms, coordinates))
    for noise in config["noise_levels"]:
        for sampling, arms, coordinates in conditions:
            condition = f"{sampling}_L{arms}_noise{noise:g}"
            np.save(
                root / "acquisition" / (f"{sampling}_L{arms}_coordinates.npy"),
                coordinates,
            )
            for name, (series, dictionary) in prepared.items():
                print("Acquiring", name, condition, flush=True)
                acquisition = acquire_image_series(
                    series.images,
                    coordinates,
                    oversampling=settings.trajectory.oversampling,
                    kernel_width=settings.trajectory.kernel_width,
                    noise_std_per_channel=noise,
                    noise_seed=config["noise_seed"],
                )
                frames, gains = reconstruct_with_impulse_gain(
                    acquisition,
                    density_compensation=(
                        "none"
                        if sampling == "cartesian"
                        else settings.comparison.density_compensation
                    ),
                    unit_impulse_gain=sampling == "spiral"
                    and settings.comparison.unit_impulse_gain,
                )
                fitted = match_complex_dictionary(frames, dictionary, phantom.support)
                metrics = evaluate_parameter_maps(
                    truth, fitted.maps, phantom.support, regions
                )
                key = name + "_" + condition
                np.savez(
                    root / "acquisition" / (key + ".npz"), kspace=acquisition.kspace
                )
                np.savez(
                    root / "reconstruction" / (key + ".npz"), frames=frames, gains=gains
                )
                np.savez(
                    root / "parameter_maps" / (key + ".npz"),
                    **fitted.maps,
                    **{"error_" + k: fitted.maps[k] - v for k, v in truth.items()},
                    valid=fitted.valid,
                    ambiguous=fitted.ambiguous,
                )
                records.append(
                    dict(
                        sequence=name,
                        condition=condition,
                        sampling=sampling,
                        interleaves=arms,
                        noise=noise,
                        seed=config["noise_seed"],
                        ambiguous=int(fitted.ambiguous[phantom.support].sum()),
                        metrics=metrics,
                    )
                )
                write_json(root / "metrics" / "results.json", records)
    with (root / "metrics" / "results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sequence",
                "condition",
                "noise",
                "seed",
                "region",
                "parameter",
                "mae",
                "rmse",
                "nrmse",
                "bias",
            ]
        )
        for row in records:
            for region, parameters in row["metrics"].items():
                for parameter, metric in parameters.items():
                    writer.writerow(
                        [
                            row["sequence"],
                            row["condition"],
                            row["noise"],
                            row["seed"],
                            region,
                            parameter,
                            *[metric[k] for k in ("mae", "rmse", "nrmse", "bias")],
                        ]
                    )
    from verify_outputs import verify

    verify(root)
    from analysis import make_report

    make_report(
        root, config, truth, phantom.support, records, gates, optimization_summary
    )
    metadata["runtime_seconds"] = time.perf_counter() - start
    write_json(root / "config" / "metadata.json", metadata)
    print(root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)
