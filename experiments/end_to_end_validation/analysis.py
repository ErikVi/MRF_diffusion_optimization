"""Experiment-specific figures and reporting from saved numerical results."""

import json
import numpy as np
import matplotlib.pyplot as plt

PARAMETERS = {
    "t1_ms": "T1 (ms)",
    "t2_ms": "T2 (ms)",
    "md_mm2_per_s": "MD (mm²/s)",
    "fa": "FA (dimensionless)",
    "proton_density": "Relative proton density",
}


def save(root, name, fig):
    for suffix in ("png", "pdf"):
        fig.savefig(
            root / "figures" / (name + "." + suffix), dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


def make_report(root, config, truth, support, records, gates, optimization):
    information = json.loads(
        (root / "metrics" / "physical_parameter_information.json").read_text()
    )
    post_checks = root / "metrics" / "post_run_checks.json"
    post_summary = (
        post_checks.read_text()
        if post_checks.exists()
        else "Post-run checks have not yet been executed."
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    positions = np.arange(5)
    for i, name in enumerate(("initial", "optimized")):
        values = np.asarray(information[name]["relative_sd"][:5])
        axes[0].bar(positions + 0.35 * i, values, width=0.35, label=name)
    axes[0].set_xticks(positions + 0.175, ["T1", "T2", "density", "MD", "FA"])
    axes[0].set_ylabel("Relative SD bound (fixed orientation)")
    axes[0].legend()
    change = 100 * (
        np.array(information["optimized"]["relative_sd"][:5])
        / np.array(information["initial"]["relative_sd"][:5])
        - 1
    )
    axes[1].bar(positions, change)
    axes[1].set_xticks(positions, ["T1", "T2", "density", "MD", "FA"])
    axes[1].set_ylabel("Change in relative SD bound (%)")
    axes[1].axhline(0, color="black", lw=0.5)
    fig.suptitle(
        "Finite-difference diagnostic: current tensor model, not optimized objective"
    )
    save(root, "tensor_parameter_information_comparison", fig)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    for ax, (parameter, label) in zip(axes, PARAMETERS.items()):
        im = ax.imshow(np.where(support, truth[parameter], np.nan))
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.65)
    save(root, "ground_truth_maps", fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for name in ("initial", "optimized"):
        signals = np.load(root / "signals" / (name + ".npz"))["dictionary"][0]
        for ax, values, label in zip(
            axes.flat,
            (abs(signals), np.angle(signals), signals.real, signals.imag),
            ("Magnitude", "Signal phase (rad)", "Real", "Imaginary"),
        ):
            ax.plot(values, label=name)
            ax.set_ylabel(label)
            ax.set_xlabel("Frame index")
    axes[0, 0].legend()
    save(root, "representative_complex_fingerprints", fig)
    noiseless = [r for r in records if r["noise"] == 0]
    # Shared truth/recovery and symmetric error limits across ALL displayed conditions.
    for parameter, label in PARAMETERS.items():
        arrays = [
            np.load(
                root
                / "parameter_maps"
                / (r["sequence"] + "_" + r["condition"] + ".npz")
            )[parameter]
            for r in noiseless
        ]
        error_limit = max(
            float(np.nanmax(abs(a[support] - truth[parameter][support])))
            for a in arrays
        )
        error_limit = max(error_limit, 1e-12)
        low = float(np.min(truth[parameter][support]))
        high = float(np.max(truth[parameter][support]))
        fig, axes = plt.subplots(
            len(noiseless), 3, figsize=(8, 2 * len(noiseless)), squeeze=False
        )
        for row, actual, axs in zip(noiseless, arrays, axes):
            for ax, data, title in zip(
                axs,
                (truth[parameter], actual, actual - truth[parameter]),
                ("Ground truth", "Recovered", "Error"),
            ):
                error = title == "Error"
                im = ax.imshow(
                    np.where(support, data, np.nan),
                    vmin=-error_limit if error else low,
                    vmax=error_limit if error else high,
                    cmap="coolwarm" if error else "viridis",
                )
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, shrink=0.7)
            axs[0].set_ylabel(row["sequence"] + "\n" + row["condition"])
        fig.suptitle(label)
        save(root, parameter + "_recovery_and_error", fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (parameter, label) in zip(axes.flat, list(PARAMETERS.items())[:4]):
        for name in ("initial", "optimized"):
            for noise in config["noise_levels"]:
                rows = [
                    r
                    for r in records
                    if r["sequence"] == name
                    and r["sampling"] == "spiral"
                    and r["noise"] == noise
                ]
                ax.plot(
                    [r["interleaves"] for r in rows],
                    [r["metrics"]["all"][parameter]["rmse"] for r in rows],
                    marker="o",
                    label=f"{name}, σ={noise:g}",
                )
        ax.set(xlabel="Spiral interleaves/frame", ylabel=label + " RMSE")
    axes[0, 0].legend(fontsize=8)
    save(root, "parameter_rmse_vs_interleaves", fig)
    noisy = [r for r in records if r["noise"] > 0 and r["sampling"] == "spiral"]
    if noisy:
        condition = next(
            r["condition"]
            for r in noisy
            if r["interleaves"]
            == config["interleaf_counts"][len(config["interleaf_counts"]) // 2]
        )
        recovered = {
            name: np.load(root / "parameter_maps" / (name + "_" + condition + ".npz"))
            for name in ("initial", "optimized")
        }
        fig, axes = plt.subplots(4, 5, figsize=(15, 11), layout="constrained")
        for axs, (parameter, label) in zip(axes, list(PARAMETERS.items())[:4]):
            initial_map, optimized_map = (
                recovered["initial"][parameter],
                recovered["optimized"][parameter],
            )
            errors = [initial_map - truth[parameter], optimized_map - truth[parameter]]
            error_limit = max(float(np.nanmax(abs(error[support]))) for error in errors)
            error_limit = max(error_limit, 1e-12)
            for index, (ax, data, title) in enumerate(
                zip(
                    axs,
                    [truth[parameter], initial_map, optimized_map, *errors],
                    [
                        "Truth",
                        "Initial recovery",
                        "Optimized recovery",
                        "Initial error",
                        "Optimized error",
                    ],
                )
            ):
                error = index >= 3
                im = ax.imshow(
                    np.where(support, data, np.nan),
                    vmin=(
                        -error_limit
                        if error
                        else float(np.min(truth[parameter][support]))
                    ),
                    vmax=(
                        error_limit
                        if error
                        else float(np.max(truth[parameter][support]))
                    ),
                    cmap="coolwarm" if error else "viridis",
                )
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                colorbar = fig.colorbar(im, ax=ax, shrink=0.65)
                if parameter == "md_mm2_per_s":
                    colorbar.formatter.set_powerlimits((-3, 3))
                    colorbar.update_ticks()
                if error and error_limit == 1e-12:
                    colorbar.set_ticks([0.0])
            axs[0].set_ylabel(label)
        fig.suptitle(condition + ": common display scales per parameter")
        save(root, "noisy_parameter_recovery_comparison", fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, arms in zip(axes, config["interleaf_counts"]):
        coords = np.load(root / "acquisition" / f"spiral_L{arms}_coordinates.npy")[0]
        for arm in coords:
            ax.plot(arm[:, 1], arm[:, 0], linewidth=0.4)
        ax.set(
            title=f"{arms} interleaves",
            xlabel="kx (cycles/pixel)",
            ylabel="ky (cycles/pixel)",
            aspect="equal",
        )
    save(root, "spiral_sampling_geometry", fig)
    example = f"spiral_L{config['interleaf_counts'][0]}_noise0"
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for axs, name in zip(axes, ("initial", "optimized")):
        frames = np.load(root / "reconstruction" / (name + "_" + example + ".npz"))[
            "frames"
        ]
        samples = np.load(root / "acquisition" / (name + "_" + example + ".npz"))[
            "kspace"
        ]
        for ax, data, title in zip(
            axs,
            (abs(frames[0]), np.angle(frames[0]), np.log10(abs(samples[0]) + 1e-12)),
            ("Reconstructed magnitude", "Reconstructed phase (rad)", "log10 |k-space|"),
        ):
            im = ax.imshow(data, aspect="auto")
            ax.set_title(name + ": " + title)
            fig.colorbar(im, ax=ax)
    save(root, "representative_acquisition_and_frames", fig)
    certification = {
        name: dict(status=status, reason=reason)
        for name, status, reason in [
            (
                "EPG signal simulation",
                "PARTIAL",
                "Characterization passes; existing gradient/tensor defects remain.",
            ),
            (
                "RF phase handling",
                "PASS",
                "Actual decoded paired phase trains used once in simulator and identical dictionaries.",
            ),
            (
                "diffusion tensor simulation",
                "PARTIAL",
                "D01/D02 attenuation discrepancies and wavevector calibration unresolved.",
            ),
            (
                "MD phantom generation",
                "PASS",
                "Tensor-derived trace/3 from positive prolate tensors.",
            ),
            (
                "FA phantom generation",
                "PASS",
                "Eigenvalue invariant of valid prolate tensors.",
            ),
            (
                "trajectory generation",
                "PARTIAL",
                "Numerical geometry validated; analytic spiral lacks scanner certification.",
            ),
            (
                "density compensation",
                "PARTIAL",
                "Explicit radial-increment heuristic; not a universal inverse.",
            ),
            (
                "NUFFT acquisition",
                "PASS",
                "Prior exact-DFT/external common-case validation and Cartesian gate.",
            ),
            (
                "reconstruction",
                "PARTIAL",
                "Cartesian sanity passes; weighted adjoint leaves spiral artifacts.",
            ),
            (
                "dictionary/model matching",
                "PARTIAL",
                "Exact on-grid reference recovery; restricted dictionary, no off-grid certification.",
            ),
            *[
                (
                    p + " recovery",
                    "PARTIAL",
                    "Noiseless Cartesian recovery passes; spiral performance conditional on grid and legacy model.",
                )
                for p in ("T1", "T2", "MD", "FA")
            ],
            (
                "optimized-phase sequence comparison",
                "PARTIAL",
                "Joint change cannot isolate causal phase benefit; objective excludes diffusion.",
            ),
            (
                "complete end-to-end pipeline",
                "PARTIAL",
                "Computational experiment completed; physical diffusion certification remains unresolved.",
            ),
        ]
    }
    (root / "report" / "certification.json").write_text(
        json.dumps(certification, indent=2)
    )
    table = [
        "| Condition | Parameter | Initial RMSE | Optimized RMSE |",
        "|---|---|---:|---:|",
    ]
    information_table = [
        "| Parameter | Initial relative SD | Optimized relative SD |",
        "|---|---:|---:|",
    ]
    for index, parameter in enumerate(information["initial"]["parameter_order"][:5]):
        information_table.append(
            f"| {parameter} | {information['initial']['relative_sd'][index]:.8g} | {information['optimized']['relative_sd'][index]:.8g} |"
        )
    outcomes = {
        parameter: {"lower": 0, "equal": 0, "higher": 0}
        for parameter in list(PARAMETERS)[:4]
    }
    for row in records:
        if row["sequence"] != "initial":
            continue
        other = next(
            r
            for r in records
            if r["sequence"] == "optimized" and r["condition"] == row["condition"]
        )
        for parameter in PARAMETERS:
            table.append(
                f"| {row['condition']} | {parameter} | {row['metrics']['all'][parameter]['rmse']:.8g} | {other['metrics']['all'][parameter]['rmse']:.8g} |"
            )
            if row["sampling"] == "spiral" and parameter in outcomes:
                difference = (
                    other["metrics"]["all"][parameter]["rmse"]
                    - row["metrics"]["all"][parameter]["rmse"]
                )
                outcomes[parameter][
                    (
                        "lower"
                        if difference < -1e-12
                        else "higher" if difference > 1e-12 else "equal"
                    )
                ] += 1
    outcome_summary = "; ".join(
        f"{parameter}: lower {values['lower']}, equal {values['equal']}, higher {values['higher']}"
        for parameter, values in outcomes.items()
    )
    report = f"""# End-to-end diffusion-MRF experiment: {config['run_name']}

## Objective
Compare an educated starting sequence with joint flip-angle/RF-phase optimization under controlled image-domain undersampling. This is a model-consistency experiment, not scanner validation.

## Initial sequence
Smooth sin² flip-angle arch: {config['initial_minimum_rad']} + {config['initial_excursion_rad']} sin²(π n/(N−1)) radians, {config['length']} readouts. A smooth arch varies saturation while respecting the established 0.02-rad step constraint; it is an educated design, not a recovered MSc artifact. Historical initial arrays are unavailable. The existing quadratic phase initializer uses fraction {config['initial_phase_fraction']}. Actual decoded spline trains, rather than the target arch, define the baseline. The known spline fitting-grid discrepancy is preserved. All exact arrays and coefficients are saved.

## Optimization
Existing SLSQP, cubic spline angle and integrated-curvature phase parameterization, L1 weighted relative SD objective, existing coefficient/angle/step constraints. Two representative tensors/tissues have equal weight. No objective change. Initial objective **{optimization['initial_objective']:.9g}**, final **{optimization['final_objective']:.9g}**. Success={optimization['success']}; {optimization['message']}; {optimization['iterations']} iterations; {optimization['runtime_seconds']:.2f} seconds including derivative compilation. SciPy default SLSQP ftol=1e−6; iteration cap {config['maximum_iterations']}. See resolved configuration, parameter archive and history.

## Optimized sequence
Both flip-angle and RF-phase coefficients are optimized. Exact paired trains and their comparison are saved. No flip-only optimum was available; no causal phase-only conclusion is possible.

## Information-theoretic results
Established objective and its relative SD bounds condition on fixed diffusion and cover T1/T2/internal equilibrium M only. Legacy MD/FA derivatives are invalid. A separate central finite-difference diagnostic perturbs the valid prolate tensor family and fits a constant object-phase nuisance parameter. Its step-halving check, full Jacobian, raw FIM, scaled conditioning and relative SD bounds are saved. This uses the same relative-SD normalization and per-channel sigma 10^(−1.65); it does not replace the objective or certify the underlying diffusion physics. Internal M changes recovery while initial Z0 remains 1; fitted image proton density instead scales the entire fingerprint. These are distinct parameters.

Diagnostic tissue: T1=750 ms, T2=70 ms, density=1, MD=0.0007 mm²/s, FA=0.2, known principal axis x. Object phase is an estimated nuisance; orientation is fixed. This continuous local diagnostic is distinct from discrete matching.

{chr(10).join(information_table)}

Scaled FIM ranks: {information['initial']['rank']} and {information['optimized']['rank']}; condition numbers {information['initial']['scaled_condition']:.6g} and {information['optimized']['scaled_condition']:.6g}. Derivative step-halving relative differences: {information['initial']['derivative_step_convergence']:.3g} and {information['optimized']['derivative_step_convergence']:.3g}. Full numerical details are in `metrics/physical_parameter_information.json` and `signals/*_physical_parameter_information.npz`.

## Phantom
{config['image_size']}×{config['image_size']} checkerboard with {int(np.count_nonzero(support))} active pixels (the default includes the full rectangular field of view): T1 750/1250 ms, T2 70/90 ms, density 1/0.9, MD 0.0007/0.001 mm²/s, FA 0.2/0.7, principal axes x/y. Prolate tensors use a=FA/sqrt(3−2FA²), eigenvalues MD(1+2a), MD(1−a), MD(1−a). FA/MD are derived from tensors. The two compartments correlate all parameters, limiting generalization.

## Acquisition
Three diffusion directions, TR=15 ms, TE=4 ms, existing preparation and legacy wavevectors. {config['state_count']} EPG states. Complex signals retain RF phase; optional static object phase is separately multiplied once, edge {config['object_phase_edge_rad']} rad. Analytic SigPy variable-density spirals, negative golden-angle schedule, interleaves {config['interleaf_counts']}; Cartesian complete-grid reference. Coordinate order ky/kx, cycles/pixel. SigPy oversampling 2, kernel width 6. See resolved configuration for FOV/gradient/slew assumptions. No readout-time off-resonance or coil model.

## Reconstruction
Cartesian adjoint; spiral radial-increment density weighting followed by adjoint and center impulse gain normalization. No iterative inversion. Restricted dictionary has 2×2×2×2×2=32 tissue/tensor entries; density and constant complex object phase are fitted analytically. Same grid, phantom, trajectories and Gaussian noise samples for both sequences. Noise SD per real/imaginary k-space channel {config['noise_levels']}, seed {config['noise_seed']}; fixed absolute SD, not fixed SNR. Noise-free conditions execute first. No claim of population-level noise statistics from one seed.

## Validation
Both reference gates passed: {json.dumps(gates)}. All dictionary atoms self-match, direct phantom matches, and Cartesian complex acquisition/reconstruction recovers known on-grid parameters. Existing independent tests cover homogeneous and contrast cases. This run does not add off-grid or scanner validation.

Pre-experiment tests: 170 passed and 9 expected failures in the existing suite, plus one passing new information-diagnostic test. The nine known failures were not suppressed or reclassified. Final test XML is supplied with the results when available.

Post-run state-count and paired-noise diagnostics: {post_summary}

## Undersampling results
NRMSE = RMSE / RMS ground truth inside each mask; global and compartment metrics saved separately. No heterogeneous parameter score is used.

{chr(10).join(table)}

## Effect of phase optimization
The paired sequence comparison changes flip angles and RF phases together. Lower objective does not guarantee lower MD/FA error, nor isolate an RF-phase effect. Consult each sampling/noise row; do not infer superiority beyond those cases.

Across tested spiral/noise conditions, optimized versus initial RMSE counts are: **{outcome_summary}**. These are descriptive counts, not independent statistical trials. The successful Cartesian gates and identical noise checks rule out those particular implementation mismatches in this comparison. The observed degradation is consistent with a signal-level independent-noise objective failing to penalize spatially structured aliasing, coupled to discrete matching. This interpretation does not prove a unique causal mechanism or rule out every remaining model limitation.

## Limitations
Known tensor attenuation D01/D02 and gradient/model conventions are unchanged. Legacy MD trace and metric derivatives cannot certify MD/FA information. Discrete 32-entry on-grid dictionary is deliberately phantom-specific; orientation family is restricted. Weighted adjoint is not an inverse. Spiral generation is not hardware-certified. Prior external UEE rotating-case disagreement remains: analytical linearized UEE and nonlinear matching are different estimators. Heesterbeek's GPL-3.0 repository at commit 4a7c0eddb8be2a6ce8a36c97a01fafcf13e7665e is a methodological reference; no external source was copied in this experiment. Our acquisition methodology was independently implemented and cross-validated; EPG remains this project's implementation. No new tensor-physics certification follows from on-grid recovery. Smoke runs are orchestration checks only.

## Certification
{chr(10).join('- '+k+': **'+v['status']+'** — '+v['reason'] for k,v in certification.items())}

## Conclusion
This run measures conditional reconstruction performance of two sequences in the same simulator. The objective changed by {100*(optimization['final_objective']/optimization['initial_objective']-1):.3f}%. Image-domain outcomes are parameter- and condition-dependent, as the table and counts above show; an objective improvement cannot establish broad undersampling superiority. This does not certify the unresolved diffusion physics. All numerical metrics, gains, raw k-space, complex images, recovered/error maps and configuration are retained for independent inspection.
"""
    (root / "report" / "experiment_report.md").write_text(report, encoding="utf-8")
