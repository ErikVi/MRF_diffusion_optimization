"""Regenerate the chronological scientific report exclusively from saved outputs."""

from pathlib import Path
import argparse
import csv
import html
import json
import hashlib
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from output import write_json

PARAMETERS = {
    "t1_ms": ("T1", "ms", 1.0),
    "t2_ms": ("T2", "ms", 1.0),
    "md_mm2_per_s": ("MD", "10⁻³ mm²/s", 1000.0),
    "fa": ("FA", "dimensionless", 1.0),
    "proton_density": ("Proton density", "relative", 1.0),
}
NAMES = ("initial", "flip_only", "final")
LABELS = ("Initial", "Flip-angle only", "Final selected phase")
COLORS = ("#285f9e", "#d17a16", "#19866b")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(root, stage, name, figure):
    figure.savefig(root / stage / (name + ".png"), dpi=250, bbox_inches="tight")
    figure.savefig(root / stage / (name + ".svg"), bbox_inches="tight")
    plt.close(figure)
    return f"../{stage}/{name}.png"


def table(headers, rows):
    def fmt(x):
        return f"{x:.6g}" if isinstance(x, (float, np.floating)) else str(x)

    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "\n".join("| " + " | ".join(fmt(x) for x in row) + " |" for row in rows)
    )


def html_report(markdown):
    """Small renderer for this report's headings, tables and local image links."""
    lines = markdown.splitlines()
    output = []
    in_table = False
    for line in lines:
        if line.startswith("| "):
            cells = [x.strip() for x in line.strip("|").split("|")]
            if all(x == "---" for x in cells):
                continue
            if not in_table:
                output.append("<table>")
                in_table = True
            output.append(
                "<tr>"
                + "".join("<td>" + html.escape(x) + "</td>" for x in cells)
                + "</tr>"
            )
            continue
        if in_table:
            output.append("</table>")
            in_table = False
        match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if match:
            output.append(
                f'<figure><img src="{html.escape(match[2])}" alt="{html.escape(match[1])}"><figcaption>{html.escape(match[1])}</figcaption></figure>'
            )
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            output.append(f"<h{level}>{html.escape(line[level:].strip())}</h{level}>")
        elif line.strip():
            output.append("<p>" + html.escape(line) + "</p>")
    if in_table:
        output.append("</table>")
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete diffusion-MRF example</title><style>body{max-width:1200px;margin:40px auto;font:17px/1.55 system-ui;color:#172536;padding:20px}img{max-width:100%}table{border-collapse:collapse;font-size:14px}td{border:1px solid #ccd;padding:7px}tr:first-child{background:#e9eef5;font-weight:bold}figure{margin:24px 0}h2{margin-top:45px}figcaption{color:#536477}</style><body>'
        + "\n".join(output)
        + "</body></html>"
    )


def make_report(root, reuse_figures=False):
    root = Path(root)

    def save(root, stage, name, figure):
        if (
            reuse_figures
            and (root / stage / (name + ".png")).is_file()
            and (root / stage / (name + ".svg")).is_file()
        ):
            plt.close(figure)
            return f"../{stage}/{name}.png"
        return save_figure(root, stage, name, figure)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    resolved = read_json(root / "data/resolved.json")
    config = resolved["experiment"]
    rows = list(csv.DictReader((root / "summary.csv").open(encoding="utf-8")))
    for row in rows:
        for k in row:
            if k not in ("name", "stage", "status"):
                row[k] = float(row[k]) if row[k] else np.nan
    lookup = {r["name"]: r for r in rows}
    summary = read_json(root / "02_flip_angle_optimization/optimization.json")
    refinement = read_json(root / "04_final_sequence/refinement.json")
    records = read_json(root / "11_final_comparison/results.json")
    gates = read_json(root / "09_reconstruction/reference_gates.json")
    truth = np.load(root / "07_phantom/ground_truth.npz")
    stages = ("01_initial_sequence", "02_flip_angle_optimization", "04_final_sequence")
    seq = {n: np.load(root / s / (n + ".npz")) for n, s in zip(NAMES, stages)}
    figures = {}

    def trains(key, stage, names, components, title):
        fig, axes = plt.subplots(
            len(components),
            1,
            figsize=(9, 2.6 * len(components)),
            squeeze=False,
            sharex=True,
        )
        for ax, (component, label) in zip(axes[:, 0], components):
            for name in names:
                i = NAMES.index(name)
                ax.plot(seq[name][component], label=LABELS[i], color=COLORS[i])
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
        axes[-1, 0].set_xlabel("Readout index within each diffusion direction")
        axes[0, 0].legend()
        fig.suptitle(title)
        fig.tight_layout()
        figures[key] = save(root, stage, key, fig)

    components = [
        ("flip_angles_rad", "Flip angle (rad)"),
        ("rf_phases_rad", "RF phase (rad)"),
    ]
    trains(
        "initial_flip_angles",
        "01_initial_sequence",
        ["initial"],
        components[:1],
        f"Educated smooth initialization — {config['length']} readouts/direction",
    )
    trains(
        "initial_rf_phases",
        "01_initial_sequence",
        ["initial"],
        components[1:],
        "Controlled initial RF phase: zero",
    )
    trains(
        "initial_sequence_overview",
        "01_initial_sequence",
        ["initial"],
        components,
        "Initial sequence; TE=4 ms, TR=15 ms",
    )
    trains(
        "flip_angle_optimization",
        "02_flip_angle_optimization",
        ["initial", "flip_only"],
        components[:1],
        "Flip-angle optimization with zero RF phase",
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        [0] + [h["iteration"] for h in summary["history"]],
        [summary["initial_objective"]] + [h["objective"] for h in summary["history"]],
        ".-",
    )
    ax.set(
        xlabel="SLSQP iteration",
        ylabel="Established weighted relative-SD objective",
        title="Flip-angle optimization history",
    )
    figures["history"] = save(
        root, "02_flip_angle_optimization", "optimization_history", fig
    )
    candidates = [r for r in rows if r["name"].startswith("candidate_")]
    fig, axes = plt.subplots(
        len(candidates), 1, figsize=(9, 2 * len(candidates)), sharex=True
    )
    for ax, row in zip(axes, candidates):
        family = row["name"].removeprefix("candidate_")
        projection = np.load(
            root / "03_phase_method_comparison" / (family + "_projection.npz")
        )
        ax.plot(
            projection["raw_rf_phases_rad"],
            color=".65",
            linestyle="--",
            label="Generator",
        )
        ax.plot(projection["applied_rf_phases_rad"], label="Applied spline")
        ax.set_ylabel(f"{family}\nrad")
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("Readout index")
    figures["phase_candidates"] = save(
        root, "03_phase_method_comparison", "phase_candidates_raw_and_applied", fig
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        [r["name"].removeprefix("candidate_") for r in candidates],
        [r["objective"] for r in candidates],
    )
    ax.set(
        ylabel="Established objective",
        title="Pre-phantom phase-family selection (lower is better)",
    )
    figures["phase_objectives"] = save(
        root, "03_phase_method_comparison", "phase_method_objectives", fig
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    for row in candidates:
        family = row["name"].removeprefix("candidate_")
        scan = list(
            csv.DictReader(
                (root / "03_phase_method_comparison" / (family + "_scan.csv")).open()
            )
        )
        ax.plot(
            [float(x["fraction"]) for x in scan],
            [float(x["objective"]) if x["objective"] else np.nan for x in scan],
            ".-",
            label=family,
        )
    ax.set(
        xlabel="Generator fraction",
        ylabel="Established objective",
        title="Finite fraction-grid search; discontinuities retained",
    )
    ax.legend(ncol=3)
    figures["phase_scan"] = save(
        root, "03_phase_method_comparison", "phase_fraction_search", fig
    )

    def information_plot(key, stage, names):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        groups = [
            (
                ["conditional_T1", "conditional_T2", "conditional_internal_M"],
                ["T1", "T2", "internal M"],
                "Objective bounds: tissue-weighted, conditional",
            ),
            (
                [
                    "diagnostic_T1",
                    "diagnostic_T2",
                    "diagnostic_PD",
                    "diagnostic_MD",
                    "diagnostic_FA",
                ],
                ["T1", "T2", "density", "MD", "FA"],
                "Physical-coordinate diagnostic: tissue 0",
            ),
        ]
        for ax, (keys, labels, title) in zip(axes, groups):
            for i, name in enumerate(names):
                ax.plot(
                    np.arange(len(keys)),
                    [lookup[name][k] for k in keys],
                    "o-",
                    label=name.removeprefix("candidate_"),
                )
            ax.set_xticks(np.arange(len(keys)), labels)
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_ylabel("Relative SD bound (log scale)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        figures[key] = save(root, stage, key, fig)

    information_plot(
        "initial_vs_flip_information",
        "02_flip_angle_optimization",
        ["initial", "flip_only"],
    )
    information_plot(
        "phase_method_information",
        "03_phase_method_comparison",
        [r["name"] for r in candidates],
    )
    fig, axes = plt.subplots(
        len(candidates), 2, figsize=(11, 1.9 * len(candidates)), sharex=True
    )
    for pair, row in zip(axes, candidates):
        signal = np.load(root / "06_signal_validation" / (row["name"] + ".npz"))[
            "fingerprints"
        ][0]
        pair[0].plot(abs(signal))
        pair[1].plot(np.angle(signal))
        pair[0].set_ylabel(row["name"].removeprefix("candidate_"))
    axes[0, 0].set_title("Magnitude (relative signal)")
    axes[0, 1].set_title("Signal phase (rad; wrapped)")
    for ax in axes[-1]:
        ax.set_xlabel("Concatenated frame (x, y, z blocks)")
    figures["candidate_signals"] = save(
        root, "03_phase_method_comparison", "phase_candidate_complex_signals", fig
    )
    trains(
        "initial_vs_final_flip_angles",
        "04_final_sequence",
        ["initial", "final"],
        components[:1],
        "Initial versus final flip angles",
    )
    trains(
        "initial_vs_final_rf_phases",
        "04_final_sequence",
        ["initial", "final"],
        components[1:],
        "Actual RF phases; applied once in EPG",
    )
    trains(
        "final_sequence_overview",
        "04_final_sequence",
        ["final"],
        components,
        "Selected spline-projected " + refinement["final"]["family"] + " phase family",
    )
    information_plot("three_stage_information", "05_information_validation", NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    keys = [
        "diagnostic_T1",
        "diagnostic_T2",
        "diagnostic_PD",
        "diagnostic_MD",
        "diagnostic_FA",
    ]
    for name, color in zip(NAMES[1:], COLORS[1:]):
        axes[0].plot(
            range(5),
            [100 * (lookup[name][k] / lookup["initial"][k] - 1) for k in keys],
            "o-",
            label=name,
            color=color,
        )
    axes[0].set_xticks(range(5), ["T1", "T2", "density", "MD", "FA"])
    axes[0].axhline(0, color=".5", lw=1)
    axes[0].set_ylabel("Diagnostic relative-SD change (%)")
    axes[0].legend()
    axes[1].plot(LABELS, [lookup[n]["objective"] for n in NAMES], "o-")
    axes[1].set_ylabel("Established objective")
    axes[1].tick_params(axis="x", labelsize=8)
    figures["progression"] = save(
        root, "05_information_validation", "information_and_objective_progression", fig
    )
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for name, label, color in zip(NAMES, LABELS, COLORS):
        signals = np.load(root / "06_signal_validation" / (name + ".npz"))[
            "fingerprints"
        ]
        for i, signal in enumerate(signals):
            for ax, values in zip(
                axes[i], [abs(signal), np.angle(signal), signal.real, signal.imag]
            ):
                ax.plot(values, label=label, color=color)
    for ax, title in zip(
        axes[0], ["Magnitude", "Signal phase (rad)", "Real", "Imaginary"]
    ):
        ax.set_title(title)
    for i in range(2):
        axes[i, 0].set_ylabel(f"Tissue {i}\nrelative signal")
    for ax in axes[-1]:
        ax.set_xlabel("Frame (x, y, z blocks)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    figures["fingerprints"] = save(
        root, "06_signal_validation", "representative_complex_fingerprints", fig
    )
    # Ground-truth figure has separate units/ranges for each parameter.
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8))
    for ax, (key, (label, unit, scale)) in zip(axes, PARAMETERS.items()):
        im = ax.imshow(truth[key] * scale, cmap="viridis")
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7, label=unit)
    figures["phantom"] = save(root, "07_phantom", "ground_truth_parameter_maps", fig)
    clean = [r for r in records if r["sequence"] == "initial" and r["noise"] == 0]
    fig, axes = plt.subplots(1, len(clean), figsize=(4 * len(clean), 4), squeeze=False)
    for ax, row in zip(axes[0], clean):
        coords = np.load(
            root
            / "08_acquisition"
            / (f"{row['sampling']}_L{row['interleaves']}_coordinates.npy")
        )[0]
        for arm in coords:
            ax.plot(arm[:, 1], arm[:, 0], lw=0.6)
        ax.set(
            title=row["sampling"] + f" L={row['interleaves']}",
            xlabel="kx (cycles/pixel)",
            ylabel="ky (cycles/pixel)",
            aspect="equal",
            xlim=(-0.52, 0.52),
            ylim=(-0.52, 0.52),
        )
    fig.tight_layout()
    figures["trajectories"] = save(
        root, "08_acquisition", "sampling_trajectory_examples", fig
    )
    chosen = next(r for r in clean if r["sampling"] == "spiral")
    condition = chosen["condition"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    samples = [
        np.load(root / "08_acquisition" / (n + "_" + condition + ".npz"))["kspace"][0]
        for n in NAMES
    ]
    vmax = max(abs(x).max() for x in samples)
    for ax, label, value in zip(axes, LABELS, samples):
        im = ax.imshow(abs(value), aspect="auto", vmin=0, vmax=vmax)
        ax.set(title=label, xlabel="Sample index", ylabel="Interleaf")
        fig.colorbar(im, ax=ax, label="|k-space| (relative)")
    fig.tight_layout()
    figures["kspace"] = save(
        root, "08_acquisition", "representative_kspace_magnitude", fig
    )
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    frames = [
        np.load(root / "09_reconstruction" / (n + "_" + condition + ".npz"))["frames"][
            0
        ]
        for n in NAMES
    ]
    vmax = max(abs(x).max() for x in frames)
    for i, (label, value) in enumerate(zip(LABELS, frames)):
        im = axes[0, i].imshow(abs(value), vmin=0, vmax=vmax)
        axes[0, i].set_title(label)
        fig.colorbar(im, ax=axes[0, i], label="Magnitude")
        im = axes[1, i].imshow(
            np.angle(value), vmin=-np.pi, vmax=np.pi, cmap="twilight"
        )
        fig.colorbar(im, ax=axes[1, i], label="Signal + object phase (rad)")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    figures["frames"] = save(
        root, "09_reconstruction", "representative_complex_adjoint_frames", fig
    )
    # Every noiseless sampling condition appears; errors use a common scale across
    # all sequences AND sampling levels, including the exact Cartesian reference.
    for key, (label, unit, scale) in PARAMETERS.items():
        fig, axes = plt.subplots(
            len(clean), 7, figsize=(19, 2.7 * len(clean)), squeeze=False
        )
        recovered = [
            [
                np.load(
                    root / "10_parameter_maps" / (n + "_" + r["condition"] + ".npz")
                )[key]
                * scale
                for n in NAMES
            ]
            for r in clean
        ]
        actual = truth[key] * scale
        vmin = float(actual.min())
        vmax = float(actual.max())
        limit = max(
            float(np.nanmax(abs(v - actual))) for group in recovered for v in group
        )
        if limit == 0:
            limit = max(vmax - vmin, abs(vmax), 1.0) * 1e-12
        for i, (row, group) in enumerate(zip(clean, recovered)):
            for j, value in enumerate([actual] + group):
                im = axes[i, j].imshow(value, cmap="viridis", vmin=vmin, vmax=vmax)
            for j, value in enumerate(group, 4):
                err = axes[i, j].imshow(
                    value - actual, cmap="RdBu_r", vmin=-limit, vmax=limit
                )
            axes[i, 0].set_ylabel(row["sampling"] + f" L={row['interleaves']}")
        for ax, title in zip(
            axes[0],
            ["Truth", *LABELS, "Initial error", "Flip-only error", "Final error"],
        ):
            ax.set_title(title, fontsize=9)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes[:, :4].ravel().tolist(), shrink=0.65, label=unit)
        fig.colorbar(
            err,
            ax=axes[:, 4:].ravel().tolist(),
            shrink=0.65,
            label="Error (" + unit + ")",
        )
        fig.suptitle(
            label + " — all noiseless sampling conditions; fixed display ranges"
        )
        figures[key + "_maps"] = save(
            root, "10_parameter_maps", key + "_recovery_and_errors", fig
        )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, (label, unit, scale)) in zip(axes.ravel(), PARAMETERS.items()):
        for name, display, color in zip(NAMES, LABELS, COLORS):
            for noise in config["noise_levels"]:
                selected = sorted(
                    [
                        r
                        for r in records
                        if r["sequence"] == name
                        and r["sampling"] == "spiral"
                        and r["noise"] == noise
                    ],
                    key=lambda x: x["interleaves"],
                )
                ax.plot(
                    [r["interleaves"] for r in selected],
                    [r["metrics"]["all"][key]["rmse"] * scale for r in selected],
                    "o-" if noise == 0 else "x--",
                    color=color,
                    label=display + (" noiseless" if noise == 0 else f" SD={noise:g}"),
                )
        ax.set(
            title=label,
            xlabel="Spiral interleaves per frame",
            ylabel="RMSE (" + unit + ")",
            xticks=config["interleaf_counts"],
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    figures["rmse"] = save(
        root, "11_final_comparison", "parameter_rmse_vs_interleaves", fig
    )
    write_json(root / "report/figures.json", figures)
    certification = [
        (
            "Initial sequence",
            "PASS",
            "Actual decoded train meets constraints; baseline saved before optimization.",
        ),
        (
            "Flip-angle optimization",
            "PASS" if summary["success"] else "PARTIAL",
            summary["message"] + "; local solution only.",
        ),
        (
            "Phase candidate generation",
            "PASS",
            "Existing generators/projection; identical angle coefficients verified.",
        ),
        (
            "Phase method comparison",
            "PARTIAL",
            "Predeclared objective and finite grid; no continuous/global family optimum claim.",
        ),
        (
            "Final phase optimization",
            "PARTIAL",
            "Selected-family local grid refinement only; no free-form or joint refinement.",
        ),
        (
            "EPG simulation",
            "PARTIAL",
            "Core tests pass; preserved G01 and tensor attenuation defects remain.",
        ),
        (
            "Diffusion tensor handling",
            "PARTIAL",
            "PSD construction passes; D01/D02 signal attenuation defects remain.",
        ),
        (
            "MD calculation",
            "PASS",
            "Physical trace(D)/3 used in phantom/fitting; legacy helper excluded.",
        ),
        (
            "FA calculation",
            "PASS",
            "Eigenvalue invariant agrees with requested prolate tensor FA.",
        ),
        (
            "CRLB evaluation",
            "PARTIAL",
            "Objective excludes MD/FA; physical-coordinate FD diagnostic is conditional on legacy physics.",
        ),
        (
            "Phantom generation",
            "PASS",
            "On-grid two-compartment parameter/tensor checks passed.",
        ),
        (
            "Reference reconstruction",
            "PASS" if all(g["passed"] for g in gates.values()) else "FAIL",
            "All-atom, direct and noiseless Cartesian gates for all three sequences.",
        ),
        (
            "Spiral trajectory",
            "PARTIAL",
            "Coordinate tests/cross-validation pass; scanner timing and gradient calibration untested.",
        ),
        (
            "NUFFT acquisition",
            "PASS",
            "Validated operator reused; Cartesian complex-image gate passes.",
        ),
        (
            "Undersampled reconstruction",
            "PARTIAL",
            "Weighted adjoint is not an inverse; aliasing and gain bias remain.",
        ),
        *[
            (
                p + " recovery",
                "PARTIAL",
                "Exact on-grid reference recovery; restricted grid, undersampling errors and no off-grid certification.",
            )
            for p in ("T1", "T2", "MD", "FA")
        ],
        (
            "End-to-end pipeline",
            "PARTIAL",
            "Workflow executed; computational consistency does not certify deficient diffusion physics.",
        ),
    ]
    write_json(
        root / "report/certification.json",
        [dict(component=a, status=b, reason=c) for a, b, c in certification],
    )
    progression = table(
        [
            "Stage",
            "Objective",
            "conditional T1",
            "conditional T2",
            "internal M",
            "diagnostic MD",
            "diagnostic FA",
        ],
        [
            [
                r["name"],
                r["objective"],
                r["conditional_T1"],
                r["conditional_T2"],
                r["conditional_internal_M"],
                r["diagnostic_MD"],
                r["diagnostic_FA"],
            ]
            for r in rows
        ],
    )
    metrics_table = table(
        [
            "Sampling",
            "Noise SD",
            "Parameter",
            "Initial RMSE",
            "Flip-only RMSE",
            "Final RMSE",
        ],
        [
            [
                r["condition"],
                r["noise"],
                p,
                r["metrics"]["all"][p]["rmse"],
                next(
                    v
                    for v in records
                    if v["sequence"] == "flip_only" and v["condition"] == r["condition"]
                )["metrics"]["all"][p]["rmse"],
                next(
                    v
                    for v in records
                    if v["sequence"] == "final" and v["condition"] == r["condition"]
                )["metrics"]["all"][p]["rmse"],
            ]
            for r in records
            if r["sequence"] == "initial"
            for p in PARAMETERS
        ],
    )
    sections = []

    def section(title, text, *keys):
        sections.append(
            "## "
            + title
            + "\n\n"
            + text
            + "\n\n"
            + "\n\n".join(
                "![" + k.replace("_", " ") + "](" + figures[k] + ")" for k in keys
            )
        )

    section(
        "1. Objective",
        f"This {config['run_name']} run tests a sequential worked example: educated initialization, flip-angle optimization, phase-family selection/refinement, and identical phantom acquisition/recovery. This is not the previous joint-optimization run. Smoke results are orchestration checks only; full results are the practical scientific example."
        + "\n\nExecution evidence: "
        + json.dumps(read_json(root / "00_audit/test_suite.json"))
        + ". Backend: "
        + read_json(root / "data/metadata.json")["backend"]
        + "; computation before reporting: "
        + str(
            round(
                read_json(root / "data/metadata.json")["runtime_seconds_before_report"],
                2,
            )
        )
        + " seconds. Source snapshot, configuration, versions and raw numerical arrays accompany this report.",
    )
    section(
        "2. Starting sequence",
        f"The thesis used a truncated Sommer train (section 5.2.2), whose input array is missing. Here the target is {config['initial_minimum_rad']} + {config['initial_excursion_rad']} sin²(pi*n/(N-1)) radians, N={config['length']}, fitted by the existing cubic initializer. It starts at zero RF phase. This smooth bounded arch is an educated example, not a fabricated historical three-arch train. Actual decoded arrays and target/projection are saved.",
        "initial_flip_angles",
        "initial_rf_phases",
        "initial_sequence_overview",
    )
    section(
        "3. Initial theoretical performance",
        "Baseline evaluation occurred before SLSQP. Each stage saves raw J/F/covariance/CRLB. Conditional objective bounds are tissue-weighted relative SD for T1,T2,internal M. Separate fixed-orientation finite-difference diagnostics use T1,T2,external density,MD,FA,object phase at each tissue. The diagnostic plot/summary uses tissue 0. These estimands and normalizations must not be conflated.",
    )
    section(
        "4. Flip-angle optimization",
        f"SLSQP optimized {config['knot_setting']-3} angle coefficients with zero phase. Constraints: pi/18 to pi/3 radians, adjacent step <=0.02 rad, coefficients within +/-1000. Unchanged weighted L1 relative-SD objective, equal tissue weights, default ftol=1e-6. Status: {summary['message']}; {summary['iterations']} iterations, {summary['evaluations']} evaluations, {summary['runtime_seconds']:.2f} s including compilation. Minimum constraint residual {summary['minimum_constraint']:.3g}. Nonconverged smoke iterates are retained and explicitly labeled.",
        "flip_angle_optimization",
        "history",
        "initial_vs_flip_information",
    )
    section(
        "5. Phase-modulation strategies",
        "Tested zero phase, cumulative piecewise quadratic, cumulative piecewise linear, sinusoidal and alternating generators. Their historical degree amplitudes are converted to radians and projected into the same cubic phase spline. Generated and applied arrays/projection errors are saved. Generator quadratic is different from the double-integrated spline decoder. The discontinuous malleable decoder and unrestricted free-form optimization were not used as extra generator families.",
        "phase_candidates",
    )
    section(
        "6. Phase-method comparison",
        f"Before phantom simulation, selection_rule.json fixes minimum established objective as the criterion. All angle coefficients, tissues, timing and directions remain fixed. {config['phase_search']['coarse_count']} fractions span the configured interval for each nonzero family. No image-domain outcome is used in selection. Exact ties retain the first configured candidate. The table includes diagnostic bounds but these do not select the winner.\n\n"
        + progression,
        "phase_objectives",
        "phase_scan",
        "phase_method_information",
        "candidate_signals",
    )
    section(
        "7. Selected phase method",
        f"Selected {refinement['coarse']['family']} at fraction {refinement['coarse']['fraction']:.8g}, objective {refinement['coarse']['objective']:.9g}. The selected family was refined locally using the configured denser grid including the original winning fraction. Final fraction {refinement['final']['fraction']:.8g}; objective {refinement['final']['objective']:.9g}. Selection and refinement were completed before phantom generation. This is a bounded search result, not proof of the globally best phase method.",
    )
    section(
        "8. Final optimized sequence",
        "Flip angles equal the flip-only optimum exactly. Final RF phase is the selected historical family's refined spline projection. The workflow is sequential flip-angle SLSQP followed by phase-family fraction search; no unreported joint optimization or post-simulation RF phase rotation is applied.",
        "initial_vs_final_flip_angles",
        "initial_vs_final_rf_phases",
        "final_sequence_overview",
    )
    reduction = 100 * (
        1 - lookup["final"]["objective"] / lookup["initial"]["objective"]
    )
    section(
        "9. Theoretical improvement",
        f"Established objective changes {lookup['initial']['objective']:.9g} -> {lookup['flip_only']['objective']:.9g} -> {lookup['final']['objective']:.9g}, a {reduction:.2f}% reduction from initialization. This objective omits diffusion uncertainty (I02). Separate MD/FA diagnostics differentiate the valid tensor parameterization and verify step-halving; they do not correct the diffusion signal model. Local bounds do not predict nonlinear aliasing/matching bias.",
        "three_stage_information",
        "progression",
    )
    section(
        "10. Signal behaviour",
        "All curves use the actual RF trains and the same two tissue/tensor configurations. Signal phase is shown wrapped; direction blocks are concatenated, independently initialized, not presented as a continuous acquisition clock. Doubled-state checks on two dictionary endpoints per sequence are saved separately. Complex temporal phase is preserved throughout.",
        "fingerprints",
    )
    section(
        "11. Phantom",
        f"{config['imaging']['phantom']['image_shape']} checkerboard, two compartments: T1 750/1250 ms, T2 70/90 ms, density 1/0.9, MD 0.0007/0.001 mm²/s, FA 0.2/0.7, principal axes x/y. The prolate family has a=FA/sqrt(3-2FA²), eigenvalues MD*(1+2a), MD*(1-a), MD*(1-a). Tensor invariants reproduce requested values. Object phase is a separate fixed quadratic spatial field. Identical tissue signals are simulated once per unique tuple.",
        "phantom",
    )
    section(
        "12. Reference reconstruction",
        "All three sequences pass dictionary self-matching, direct phantom matching and noiseless Cartesian forward/adjoint/matching before spiral acquisition. The unchanged complex-image relative tolerance is 1e-4; discrete T1/T2/MD/FA reference recovery must be exact to roundoff. This is on-grid consistency, not general tissue-estimation validation.\n\n"
        + table(
            ["Sequence", "Passed", "Image relative error", "Minimum dictionary margin"],
            [
                [
                    n,
                    g["passed"],
                    g["image_relative_error"],
                    g["minimum_dictionary_correlation_margin"],
                ]
                for n, g in gates.items()
            ],
        ),
    )
    section(
        "13. Undersampling",
        f"Generated variable-density spiral with golden-angle schedule, interleaves {config['interleaf_counts']}. Coordinates are (ky,kx) cycles/pixel, converted by the validated SigPy operator. Forward samples are unweighted. Reconstruction uses radial-increment DCF and operator-only central impulse gain, not phantom truth scaling. No scanner gradient/readout-time certification is implied. Noise levels {config['noise_levels']} are absolute Gaussian SD per real/imaginary k-space channel, seed {config['noise_seed']}. Noiseless conditions run first. Paired-noise checks and measured signal-RMS/noise-RMS ratios are saved; fixed noise does not mean identical SNR for different signals.",
        "trajectories",
        "kspace",
        "frames",
    )
    section(
        "14. Quantitative reconstruction",
        "Dictionary axes are independent T1 x T2 x MD x FA x orientation: 2^5=32 atoms, identical grid for all sequences. The saved budget guards memory before generation. Density and constant object phase are analytic complex least-squares estimates, with one scalar over the entire fingerprint. Maps derive from the selected valid tensor. All noiseless sampling levels appear below with identical parameter and signed-error scales across sequences and conditions. MD figure units are 10^-3 mm²/s; CSV units remain mm²/s.",
        *[p + "_maps" for p in PARAMETERS],
    )
    section(
        "15. Initial versus optimized sequence",
        "Global RMSE in native units is listed below. The master reconstruction_metrics.csv also includes MAE, RMSE, NRMSE=RMSE/RMS(truth), bias, valid/invalid voxel counts and compartment-level metrics for every sequence, sampling and noise condition.\n\n"
        + metrics_table,
        "rmse",
    )
    conclusions = []
    for r in records:
        if r["sequence"] != "initial" or r["sampling"] != "spiral":
            continue
        f = next(
            x
            for x in records
            if x["sequence"] == "final" and x["condition"] == r["condition"]
        )
        outcomes = []
        for p in list(PARAMETERS)[:4]:
            a = r["metrics"]["all"][p]["rmse"]
            b = f["metrics"]["all"][p]["rmse"]
            outcomes.append(
                PARAMETERS[p][0]
                + (
                    " tied"
                    if np.isclose(a, b, rtol=1e-10, atol=1e-15)
                    else " improved" if b < a else " worsened"
                )
            )
        conclusions.append(r["condition"] + ": " + ", ".join(outcomes) + ".")
    section(
        "16. Discussion",
        " ".join(conclusions)
        + " The flip-only condition isolates the effect of adding the selected phase at fixed flip angles within this model. This does not establish a universal causal benefit of RF-phase optimization or transfer beyond the tested grid, tissues, sampling and reconstruction. A lower objective is not evidence of universally smaller map errors.",
    )
    phase_effect = []
    for r in records:
        if r["sequence"] != "flip_only" or r["sampling"] != "spiral":
            continue
        f = next(
            x
            for x in records
            if x["sequence"] == "final" and x["condition"] == r["condition"]
        )
        for p in list(PARAMETERS)[:4]:
            phase_effect.append(
                [
                    r["condition"],
                    PARAMETERS[p][0],
                    r["metrics"]["all"][p]["rmse"],
                    f["metrics"]["all"][p]["rmse"],
                ]
            )
    sections[-1] += (
        "\n\nAt fixed optimized flip angles, the following table isolates addition of the selected phase. Native units are ms, ms, mm²/s and dimensionless FA.\n\n"
        + table(
            ["Condition", "Parameter", "Flip-only RMSE", "Final RMSE"], phase_effect
        )
    )
    section(
        "17. Limitations",
        "The model retains nine documented strict expected failures: B01, D01, D02, M01, M02, M03, G01, I01, I02. Valid tensor maps do not fix tensor EPG attenuation. Wavevectors are not scanner-calibrated. The dictionary is coarse, on-grid and restricted to prolate tensors and two axes. The diagnostic holds orientation fixed; parameter correlations and discrete matching differ from its local noise model. SLSQP is local; phase searches are finite and projection can suppress alternating structure or smear reset gaps. Golden spiral sampling lacks intra-readout decay, coils, off-resonance and measured scanner timing. DCF-weighted adjoint is not an inverse. The external Heesterbeek analytical UEE disagreement remains: acquisition agreement is not agreement of its error predictor or general diffusion validation. No external source code was copied here; existing independently reimplemented/conceptually inspired acquisition methodology and GPL provenance are documented in docs/undersampling_validation.md.\n\n"
        + table(["Component", "Status", "Evidence / limitation"], certification),
    )
    section(
        "18. Conclusion",
        f"The worked example completes the requested computational chain with an objective reduction of {reduction:.2f}%, under the established conditional objective. Reference recovery is validated for the tested on-grid phantom. Undersampling conclusions are parameter- and condition-specific as reported above. Overall scientific certification remains PARTIAL; this run does not resolve the known diffusion model defects.",
    )
    report = (
        "# Complete diffusion-MRF optimization example\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    (root / "report/REPORT.md").write_text(report, encoding="utf-8")
    (root / "report/REPORT.html").write_text(html_report(report), encoding="utf-8")
    source = Path(__file__).read_bytes()
    (root / "report/analysis_source.py").write_bytes(source)
    write_json(
        root / "report/report_generation.json",
        dict(
            analysis_sha256=hashlib.sha256(source).hexdigest(),
            reused_existing_figures=reuse_figures,
        ),
    )
    write_json(
        root / "report/console_summary.json",
        dict(
            audit_completed=True,
            run=config["run_name"],
            figures=len(figures),
            initial_objective=lookup["initial"]["objective"],
            flip_only_objective=lookup["flip_only"]["objective"],
            selected_family=refinement["final"]["family"],
            final_objective=lookup["final"]["objective"],
            objective_improvement_percent=reduction,
            optimization_converged=summary["success"],
            reference_gates=gates,
            certification="PARTIAL",
            undersampling_findings=conclusions,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--reuse-figures",
        action="store_true",
        help="Refresh report text/tables without rewriting existing figures",
    )
    args = parser.parse_args()
    make_report(args.results, reuse_figures=args.reuse_figures)
