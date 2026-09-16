"""Post-run truncation and paired-noise checks using the exact saved sequence."""

from pathlib import Path
import argparse
import json
from dataclasses import replace
import numpy as np
from mrf_diffusion.experiments.settings import _update
from mrf_diffusion.sequence.definition import (
    SequenceSettings,
    SimulationOptions,
    MRFSequence,
)
from mrf_diffusion.simulation.tissue import TissueParameters
from mrf_diffusion.simulation.api import simulate_mrf_signal


def verify(root):
    root = Path(root)
    resolved = json.loads((root / "config/resolved.json").read_text())
    options = _update(SimulationOptions(), resolved["imaging"]["simulation"])
    physics = _update(SequenceSettings(), resolved["optimization"]["sequence"])
    results = {"state_count": {}, "paired_noise": {}}
    for name in ("initial", "optimized"):
        archive = np.load(root / "sequences" / (name + ".npz"))
        sequence = MRFSequence(
            archive["flip_angles_rad"], archive["rf_phases_rad"], physics
        )
        dictionary = np.load(root / "signals" / (name + ".npz"))
        errors = []
        for index in (0, len(dictionary["dictionary"]) - 1):
            tissue = TissueParameters(
                float(dictionary["t1_ms"][index]),
                float(dictionary["t2_ms"][index]),
                1.0,
                dictionary["tensors"][index],
            )
            components = np.asarray(
                simulate_mrf_signal(
                    tissue,
                    sequence,
                    replace(options, state_count=2 * options.state_count),
                )
            )
            actual = components[0] + 1j * components[1]
            expected = dictionary["dictionary"][index]
            errors.append(
                float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
            )
        results["state_count"][name] = dict(
            relative_errors=errors,
            passed=max(errors) < 1e-8,
            original=options.state_count,
            doubled=2 * options.state_count,
        )
    records = json.loads((root / "metrics/results.json").read_text())
    for row in records:
        if row["sequence"] != "initial" or row["noise"] == 0:
            continue
        condition = row["condition"]
        clean = condition.split("_noise")[0] + "_noise0"
        noises = []
        snrs = []
        for name in ("initial", "optimized"):
            signal = np.load(root / "acquisition" / (name + "_" + clean + ".npz"))[
                "kspace"
            ]
            noisy = np.load(root / "acquisition" / (name + "_" + condition + ".npz"))[
                "kspace"
            ]
            noises.append(noisy - signal)
            snrs.append(
                float(np.sqrt(np.mean(abs(signal) ** 2)) / (np.sqrt(2) * row["noise"]))
            )
        difference = float(np.max(abs(noises[0] - noises[1])))
        results["paired_noise"][condition] = dict(
            maximum_complex_difference=difference,
            passed=difference < 1e-12,
            expected_complex_rms_snr_initial_optimized=snrs,
        )
    (root / "metrics/post_run_checks.json").write_text(json.dumps(results, indent=2))
    if not all(v["passed"] for group in results.values() for v in group.values()):
        raise ValueError(
            "Post-run scientific diagnostic failed; inspect post_run_checks.json"
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory")
    verify(parser.parse_args().result_directory)
