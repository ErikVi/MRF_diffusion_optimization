"""Sequential flip-angle optimization, phase-family search and image validation."""

from dataclasses import asdict, replace
from pathlib import Path
import argparse
import time
import tomllib
import numpy as np
import jax
import jax.numpy as jnp
from mrf_diffusion.experiments.settings import (
    ExperimentSettings,
    TissueEnsemble,
    _update,
)
from mrf_diffusion.experiments.optimization import build_problem
from mrf_diffusion.optimization.solver import optimize_sequence
from mrf_diffusion.optimization.phase_search import (
    phase_candidate,
    search_phase_fractions,
)
from mrf_diffusion.sequence.parameterization import decode_sequence_parameters
from mrf_diffusion.sequence.definition import MRFSequence, SimulationOptions
from mrf_diffusion.diffusion.parameterization import (
    axisymmetric_tensor,
    tensor_invariants,
)
from mrf_diffusion.information.tensor_diagnostic import evaluate_tensor_information
from mrf_diffusion.simulation.signal import simulate_tensor_signal
from mrf_diffusion.experiments.quantitative_settings import QuantitativeSettings
from output import initialize_output, write_json, write_csv


def evaluate_sequence(
    root,
    stage,
    name,
    coefficients,
    points,
    knots,
    objective,
    precision,
    optimization,
    imaging,
    runtime=0.0,
    status="evaluated",
):
    """Save actual sequence and both explicitly distinct information analyses."""
    angles, phases = decode_sequence_parameters(
        coefficients, points, knots, method="free form"
    )
    sequence = MRFSequence(angles, phases, optimization.sequence)
    directory = root / stage
    np.savez(
        directory / (name + ".npz"),
        flip_angles_rad=angles,
        rf_phases_rad=phases,
        coefficients=coefficients,
    )
    for suffix, array in (("flip_angles", angles), ("rf_phases", phases)):
        np.save(directory / (name + "_" + suffix + ".npy"), array)
    np.savetxt(
        directory / (name + ".csv"),
        np.column_stack((points, angles, phases)),
        delimiter=",",
        header="readout_index,flip_angle_rad,rf_phase_rad",
        comments="",
    )
    conditional = np.asarray(precision(coefficients))
    diagnostics, signals = [], []
    for i, ((t1, t2, m), tensor) in enumerate(
        zip(
            optimization.tissues.scalar_parameters,
            optimization.tissues.diffusion_tensors,
        )
    ):
        md, fa, axis = tensor_invariants(tensor)
        params = [t1, t2, 1.0, float(md), float(fa), 0.0]
        arrays, diagnostic = evaluate_tensor_information(
            sequence,
            optimization.simulation,
            params,
            axis,
            [t1, t2, 1.0, float(md), float(fa), 1.0],
        )
        np.savez(
            root / "05_information_validation" / (name + f"_tissue{i}_diagnostic.npz"),
            **arrays,
        )
        # These AD columns are the validated subset used in the established objective.
        jac = np.asarray(
            jax.jacobian(simulate_tensor_signal, argnums=(2, 3, 4))(
                angles,
                phases,
                float(t1),
                float(t2),
                float(m),
                jnp.asarray(tensor),
                jnp.zeros(3),
                jnp.zeros(3),
                **asdict(optimization.simulation),
                sequence=sequence.settings,
            )
        )
        flat = jac.reshape(3, -1)
        fim = flat @ flat.T / 10 ** (-3.3)
        covariance = np.linalg.inv(fim)
        np.savez(
            root / "05_information_validation" / (name + f"_tissue{i}_conditional.npz"),
            jacobian=jac,
            fim=fim,
            covariance=covariance,
            crlb=np.diag(covariance),
            relative_sd=np.sqrt(np.diag(covariance)) / np.array([t1, t2, m]),
            parameter_order=np.array(["T1", "T2", "internal_M"]),
        )
        diagnostics.append(diagnostic)
        n = len(arrays["signal"]) // 2
        signals.append(arrays["signal"][:n] + 1j * arrays["signal"][n:])
    np.savez(root / "06_signal_validation" / (name + ".npz"), fingerprints=signals)
    write_json(root / "05_information_validation" / (name + ".json"), diagnostics)
    bounds = diagnostics[0]["relative_sd"]
    row = dict(
        stage=stage,
        name=name,
        objective=float(objective(coefficients)),
        conditional_T1=float(conditional[0]),
        conditional_T2=float(conditional[1]),
        conditional_internal_M=float(conditional[2]),
        diagnostic_T1=None if bounds is None else bounds[0],
        diagnostic_T2=None if bounds is None else bounds[1],
        diagnostic_PD=None if bounds is None else bounds[2],
        diagnostic_MD=None if bounds is None else bounds[3],
        diagnostic_FA=None if bounds is None else bounds[4],
        runtime_seconds=runtime,
        status=status,
    )
    return sequence, row


def run(config_path):
    start = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not jax.config.x64_enabled:
        raise ValueError("This validated example requires JAX x64")
    root, metadata = initialize_output(config_path, config)
    write_json(root / "data/status.json", dict(stage="initializing", completed=False))
    imaging = _update(QuantitativeSettings(), config["imaging"])
    options = SimulationOptions(direction_count=3, state_count=config["state_count"])
    imaging = replace(imaging, simulation=options)
    tensors = tuple(
        tuple(map(tuple, axisymmetric_tensor(md, fa, axis)))
        for md, fa, axis in zip(
            imaging.diffusion.mean_diffusivity_mm2_per_s,
            imaging.diffusion.fractional_anisotropy,
            imaging.diffusion.principal_directions_xyz,
        )
    )
    optimization = replace(
        _update(ExperimentSettings(), config["optimizer"]),
        length=config["length"],
        knot_setting=config["knot_setting"],
        simulation=options,
        tissues=TissueEnsemble(
            scalar_parameters=tuple(
                (a, b, 1.0)
                for a, b in zip(imaging.phantom.t1_ms, imaging.phantom.t2_ms)
            ),
            diffusion_tensors=tensors,
            parameter_weights=tuple(config["parameter_weights"]),
            tissue_weights=tuple(config["tissue_weights"]),
        ),
    )
    # One authoritative timing/diffusion setting for objective and both spatial models.
    imaging = replace(
        imaging, sequence=replace(imaging.sequence, physics=optimization.sequence)
    )
    write_json(
        root / "data/resolved.json",
        dict(
            experiment=config,
            imaging=asdict(imaging),
            optimization=asdict(optimization),
        ),
    )
    target = (
        config["initial_minimum_rad"]
        + config["initial_excursion_rad"]
        * np.sin(np.linspace(0, np.pi, config["length"])) ** 2
    )
    prep = jnp.zeros(3)
    initial, _, points, knots, _, _, flip_objective, _, constraints = build_problem(
        optimization, jnp.asarray(target), prep, prep, "no phase modulation"
    )
    _, _, _, _, _, _, objective, precision, _ = build_problem(
        optimization, jnp.asarray(target), prep, prep, "free form"
    )
    initial_paired, _, _ = phase_candidate(initial, points, knots, "none", 0.0)
    if min(float(np.min(c["fun"](initial))) for c in constraints) < -1e-8:
        raise ValueError("Decoded educated initialization violates constraints")
    np.savez(
        root / "01_initial_sequence/initialization.npz",
        target_flip_angles_rad=target,
        knots=knots,
        points=points,
        coefficients=initial,
    )
    sequences, rows = {}, []
    sequences["initial"], row = evaluate_sequence(
        root,
        "01_initial_sequence",
        "initial",
        initial_paired,
        points,
        knots,
        objective,
        precision,
        optimization,
        imaging,
    )
    rows.append(row)
    np.testing.assert_allclose(
        row["objective"], float(flip_objective(initial)), rtol=1e-12
    )
    write_csv(root / "summary.csv", rows)
    print("Initial evaluated before optimization:", row["objective"], flush=True)
    write_json(
        root / "data/status.json",
        dict(stage="flip_angle_optimization", completed=False),
    )
    tick = time.perf_counter()
    result, history = optimize_sequence(
        flip_objective,
        initial,
        constraints,
        optimization.solver,
        optimization.constraints,
    )
    duration = time.perf_counter() - tick
    flip_summary = dict(
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        gradient_evaluations=int(result.njev),
        runtime_seconds=duration,
        initial_objective=float(flip_objective(initial)),
        final_objective=float(result.fun),
        minimum_constraint=float(min(np.min(c["fun"](result.x)) for c in constraints)),
        history=history,
        optimizer=asdict(optimization.solver),
        constraints=asdict(optimization.constraints),
        scipy_default_ftol=1e-6,
    )
    write_json(root / "02_flip_angle_optimization/optimization.json", flip_summary)
    write_csv(root / "02_flip_angle_optimization/history.csv", history)
    np.savez(
        root / "02_flip_angle_optimization/coefficients.npz",
        initial=initial,
        optimized=result.x,
    )
    if not np.isfinite(result.fun) or flip_summary["minimum_constraint"] < -1e-6:
        raise ValueError("Invalid/infeasible optimization; retained diagnostics")
    flip_paired, _, _ = phase_candidate(result.x, points, knots, "none", 0.0)
    sequences["flip_only"], row = evaluate_sequence(
        root,
        "02_flip_angle_optimization",
        "flip_only",
        flip_paired,
        points,
        knots,
        objective,
        precision,
        optimization,
        imaging,
        duration,
        str(result.message),
    )
    rows.append(row)
    write_csv(root / "summary.csv", rows)
    print("Flip-angle optimization:", flip_summary, flush=True)
    write_json(
        root / "data/status.json",
        dict(stage="phase_method_comparison", completed=False),
    )
    selection_rule = dict(
        criterion="minimum unchanged weighted L1 T1/T2/internal-M objective",
        tie_rule="first configured family/fraction",
        phase_representation="historical generator projected into existing cubic spline",
        fixed_flip_coefficients=np.asarray(result.x).tolist(),
        selection_precedes_phantom=True,
        families=config["phase_search"]["families"],
        fraction_minimum=config["phase_search"]["minimum"],
        fraction_maximum=config["phase_search"]["maximum"],
        coarse_count=config["phase_search"]["coarse_count"],
        refinement_count=config["phase_search"]["refinement_count"],
    )
    write_json(root / "03_phase_method_comparison/selection_rule.json", selection_rule)
    fractions = np.linspace(
        config["phase_search"]["minimum"],
        config["phase_search"]["maximum"],
        config["phase_search"]["coarse_count"],
    )
    best = None
    for family in config["phase_search"]["families"]:
        coefficients, candidate, scan = search_phase_fractions(
            result.x,
            points,
            knots,
            family,
            [0.0] if family == "none" else fractions,
            objective,
        )
        write_csv(root / "03_phase_method_comparison" / (family + "_scan.csv"), scan)
        _, raw, applied = phase_candidate(
            result.x, points, knots, family, candidate["fraction"]
        )
        np.savez(
            root / "03_phase_method_comparison" / (family + "_projection.npz"),
            raw_rf_phases_rad=raw,
            applied_rf_phases_rad=applied,
        )
        _, row = evaluate_sequence(
            root,
            "03_phase_method_comparison",
            "candidate_" + family,
            coefficients,
            points,
            knots,
            objective,
            precision,
            optimization,
            imaging,
            sum(r["runtime_seconds"] for r in scan),
            "finite grid minimum",
        )
        rows.append(row)
        write_csv(root / "summary.csv", rows)
        if best is None or candidate["objective"] < best["objective"]:
            best = candidate
        print("Phase candidate", candidate, flush=True)
    write_json(root / "03_phase_method_comparison/selected.json", best)
    spacing = fractions[1] - fractions[0]
    fine = np.unique(
        np.r_[
            best["fraction"],
            np.linspace(
                max(fractions[0], best["fraction"] - spacing),
                min(fractions[-1], best["fraction"] + spacing),
                config["phase_search"]["refinement_count"],
            ),
        ]
    )
    final_coefficients, final, scan = search_phase_fractions(
        result.x,
        points,
        knots,
        best["family"],
        [0.0] if best["family"] == "none" else fine,
        objective,
    )
    write_csv(root / "04_final_sequence/fraction_refinement.csv", scan)
    write_json(
        root / "04_final_sequence/refinement.json",
        dict(
            coarse=best,
            final=final,
            strategy="fixed angles, local finite-grid refinement in selected family; not joint SLSQP",
        ),
    )
    sequences["final"], row = evaluate_sequence(
        root,
        "04_final_sequence",
        "final",
        final_coefficients,
        points,
        knots,
        objective,
        precision,
        optimization,
        imaging,
        sum(r["runtime_seconds"] for r in scan),
        "finite grid minimum",
    )
    rows.append(row)
    write_csv(root / "summary.csv", rows)
    np.testing.assert_array_equal(
        sequences["flip_only"].flip_angles, sequences["final"].flip_angles
    )
    if final["objective"] > best["objective"] + 1e-12:
        raise ValueError("Refinement lost the selected candidate")
    print("Selected and refined BEFORE phantom:", final, flush=True)
    write_json(
        root / "data/status.json",
        dict(stage="phantom_reference_and_undersampling", completed=False),
    )
    from spatial_validation import run_spatial_validation

    run_spatial_validation(root, config, imaging, optimization, sequences)
    metadata["runtime_seconds_before_report"] = time.perf_counter() - start
    write_json(root / "data/metadata.json", metadata)
    write_json(
        root / "data/status.json", dict(stage="report_generation", completed=False)
    )
    from analysis import make_report

    make_report(root)
    from verify_outputs import verify, print_summary

    print_summary(verify(root))
    metadata["runtime_seconds"] = time.perf_counter() - start
    write_json(root / "data/metadata.json", metadata)
    write_json(
        root / "data/status.json",
        dict(stage="completed", completed=True, physical_certification="PARTIAL"),
    )
    print("Completed", root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)
