"""Verify saved evidence and print the worked-example summary; no optimization."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import numpy as np
from mrf_diffusion.reconstruction.metrics import evaluate_parameter_maps
from output import write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root):
    root = Path(root).resolve()
    config = read(root / "data/resolved.json")["experiment"]
    metadata = read(root / "data/metadata.json")
    assert (
        hashlib.sha256((root / "data/config.toml").read_bytes()).hexdigest()
        == metadata["config_sha256"]
    )
    summary = list(csv.DictReader((root / "summary.csv").open(encoding="utf-8")))
    lookup = {r["name"]: r for r in summary}
    selected = read(root / "03_phase_method_comparison/selected.json")
    candidates = [r for r in summary if r["name"].startswith("candidate_")]
    assert selected["objective"] == min(float(r["objective"]) for r in candidates)
    assert selected["family"] == min(candidates, key=lambda r: float(r["objective"]))[
        "name"
    ].removeprefix("candidate_")
    assert float(lookup["final"]["objective"]) <= selected["objective"] + 1e-12
    initial = np.load(root / "01_initial_sequence/initial.npz")
    flip = np.load(root / "02_flip_angle_optimization/flip_only.npz")
    final = np.load(root / "04_final_sequence/final.npz")
    np.testing.assert_array_equal(initial["rf_phases_rad"], 0)
    np.testing.assert_array_equal(flip["rf_phases_rad"], 0)
    np.testing.assert_array_equal(flip["flip_angles_rad"], final["flip_angles_rad"])
    for r in candidates:
        data = np.load(root / "03_phase_method_comparison" / (r["name"] + ".npz"))
        np.testing.assert_array_equal(data["flip_angles_rad"], flip["flip_angles_rad"])
    truth = np.load(root / "07_phantom/ground_truth.npz")
    maps = {
        k: truth[k] for k in ("t1_ms", "t2_ms", "md_mm2_per_s", "fa", "proton_density")
    }
    records = read(root / "11_final_comparison/results.json")
    assert len(records) == 3 * (1 + len(config["interleaf_counts"])) * len(
        config["noise_levels"]
    )
    assert len({(r["sequence"], r["condition"]) for r in records}) == len(records)
    for r in records:
        saved = np.load(
            root / "10_parameter_maps" / (r["sequence"] + "_" + r["condition"] + ".npz")
        )
        calculated = evaluate_parameter_maps(
            maps, saved, truth["support"], truth["regions"]
        )
        assert calculated == r["metrics"], (r["sequence"], r["condition"])
        assert saved["valid"][truth["support"]].all()
        for key, value in maps.items():
            np.testing.assert_array_equal(saved["error_" + key], saved[key] - value)
    largest_step_error = 0.0
    for r in summary:
        information = read(root / "05_information_validation" / (r["name"] + ".json"))
        for i, info in enumerate(information):
            assert info["rank"] == 6
            assert info["derivative_step_convergence"] < 1e-5
            largest_step_error = max(
                largest_step_error, info["derivative_step_convergence"]
            )
            data = np.load(
                root
                / "05_information_validation"
                / (r["name"] + f"_tissue{i}_diagnostic.npz")
            )
            np.testing.assert_allclose(
                data["jacobian"].T @ data["jacobian"] / info["sigma_per_channel"] ** 2,
                data["fim"],
                rtol=1e-12,
                atol=1e-12,
            )
            scaled_covariance = data["covariance"] / np.outer(
                data["scales"], data["scales"]
            )
            np.testing.assert_allclose(
                scaled_covariance @ data["scaled_fim"], np.eye(6), rtol=1e-8, atol=1e-8
            )
            np.testing.assert_allclose(
                data["relative_sd"] ** 2, np.diag(scaled_covariance), rtol=1e-12
            )
    gates = read(root / "09_reconstruction/reference_gates.json")
    assert all(g["passed"] for g in gates.values())
    state_checks = read(root / "06_signal_validation/state_convergence.json")
    assert all(
        max(x["doubled_state_relative_errors"]) <= 1e-12 for x in state_checks.values()
    )
    noise = read(root / "08_acquisition/noise_checks.json")
    assert all(x["paired_noise_max_difference"] <= 1e-14 for x in noise)
    figures = read(root / "report/figures.json")
    for relative in figures.values():
        path = (root / "report" / relative).resolve()
        assert path.is_file() and path.with_suffix(".svg").is_file()
    for relative in re.findall(
        r"!\[.*?\]\((.*?)\)", (root / "report/REPORT.md").read_text(encoding="utf-8")
    ):
        assert (root / "report" / relative).is_file()
    assert (root / "report/REPORT.html").is_file()
    tests = read(root / "00_audit/test_suite.json")
    comparisons = []
    for r in records:
        if r["sequence"] != "initial":
            continue
        f = next(
            x
            for x in records
            if x["sequence"] == "final" and x["condition"] == r["condition"]
        )
        comparisons.append(
            dict(
                condition=r["condition"],
                **{
                    p: [r["metrics"]["all"][p]["rmse"], f["metrics"]["all"][p]["rmse"]]
                    for p in maps
                },
            )
        )
    result = dict(
        audit_completed=True,
        tests=tests,
        run=config["run_name"],
        verification="PASS",
        results_directory=str(root),
        report=str(root / "report/REPORT.html"),
        figures=len(figures),
        optimization_converged=read(
            root / "02_flip_angle_optimization/optimization.json"
        )["success"],
        initial_objective=float(lookup["initial"]["objective"]),
        flip_only_objective=float(lookup["flip_only"]["objective"]),
        selected_phase_method=selected["family"],
        final_objective=float(lookup["final"]["objective"]),
        objective_improvement_percent=100
        * (
            1
            - float(lookup["final"]["objective"])
            / float(lookup["initial"]["objective"])
        ),
        initial_final_relative_sd_diagnostic={
            p: [float(lookup[n]["diagnostic_" + p]) for n in ("initial", "final")]
            for p in ("T1", "T2", "MD", "FA")
        },
        initial_final_conditional_objective_bounds={
            p: [float(lookup[n]["conditional_" + p]) for n in ("initial", "final")]
            for p in ("T1", "T2", "internal_M")
        },
        initial_final_rmse=comparisons,
        largest_derivative_step_error=largest_step_error,
        maximum_paired_noise_difference=max(
            (x["paired_noise_max_difference"] for x in noise), default=0.0
        ),
        scientific_certification="PARTIAL",
        unresolved="Nine known scientific discrepancies; conditional objective omits MD/FA; finite phase grid/projection; on-grid restricted phantom; uncalibrated gradients; adjoint aliasing; analytical UEE disagreement.",
    )
    write_json(root / "report/verified_summary.json", result)
    return result


def print_summary(result):
    print(
        f"Audit complete; {result['tests']['passed']} tests passed, {result['tests']['failures']} failed, {result['tests']['skipped_or_expected_failures']} expected failures."
    )
    print(
        f"{result['run']} run: saved-output verification PASS; {result['figures']} figures; optimization converged={result['optimization_converged']}"
    )
    print("Results:", result["results_directory"])
    print("Report:", result["report"])
    print(
        f"Objective: {result['initial_objective']:.9g} -> {result['flip_only_objective']:.9g} -> {result['final_objective']:.9g}; reduction {result['objective_improvement_percent']:.2f}%"
    )
    print("Selected phase:", result["selected_phase_method"])
    print(
        "Physical-coordinate diagnostic relative SD, initial -> final (not optimized objective):"
    )
    for parameter, values in result["initial_final_relative_sd_diagnostic"].items():
        print(f"  {parameter}: {values[0]:.6g} -> {values[1]:.6g}")
    print("RMSE initial -> final, columns T1(ms), T2(ms), MD(mm2/s), FA:")
    for row in result["initial_final_rmse"]:
        print(
            row["condition"]
            + ": "
            + " | ".join(
                f"{row[p][0]:.6g} -> {row[p][1]:.6g}"
                for p in ("t1_ms", "t2_ms", "md_mm2_per_s", "fa")
            )
        )
    print("Scientific certification:", result["scientific_certification"])
    print("Unresolved:", result["unresolved"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    print_summary(verify(parser.parse_args().results))
