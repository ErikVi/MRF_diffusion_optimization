"""Regenerate figures/report using saved results; no MRI simulation or optimization."""

import argparse
import json
from pathlib import Path
import tomllib
import numpy as np
import matplotlib

matplotlib.use("Agg")
from analysis import make_report


def replot(root):
    root = Path(root).resolve()
    phantom = np.load(root / "phantom" / "ground_truth.npz")
    truth = {
        k: phantom[k]
        for k in ("t1_ms", "t2_ms", "md_mm2_per_s", "fa", "proton_density")
    }

    def read(path):
        return json.loads((root / path).read_text())

    make_report(
        root,
        tomllib.loads((root / "config" / "experiment.toml").read_text()),
        truth,
        phantom["support"],
        read("metrics/results.json"),
        read("metrics/reference_gates.json"),
        read("optimization/summary.json"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory")
    replot(parser.parse_args().result_directory)
