"""Compare supplied external helpers without importing their EPG or script setup.

Obtain the GPL-3.0 reference separately at the revision documented below.
Only named, unmodified methods are executed. This tool does not download code,
copy it into the package, or run the external optimization.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import sigpy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mrf_diffusion.encoding import (
    generate_rotated_spiral_trajectory,
    radial_increment_density_compensation,
    direct_sampling_psf,
    NufftOperator,
    calculate_point_spread_function,
)

REVISION = "4a7c0eddb8be2a6ce8a36c97a01fafcf13e7665e"
BLOB = "d28215569e2cbe74b6a1247fdc896edf87fb9bc2"


def compare(reference_directory):
    source = (reference_directory / "UEE_phase.py").read_bytes()
    # Git blob identity pins exactly the audited implementation (including CRLF).
    digest = hashlib.sha1(
        b"blob " + str(len(source)).encode() + b"\0" + source
    ).hexdigest()
    if digest != BLOB:
        raise ValueError(f"Reference differs from pinned revision {REVISION}: {digest}")
    tree = ast.parse(source)
    cost_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Cost"
    )
    names = {"Spiral_coord", "Spiral_dcf", "Coord", "P_single", "P_single_fft"}
    cost_class.body = [
        node
        for node in cost_class.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=[cost_class], type_ignores=[])
    namespace = {"np": np, "sp": sp, "tqdm": lambda iterable, **kwargs: iterable}
    exec(
        compile(
            ast.fix_missing_locations(module), "external_reference_helpers", "exec"
        ),
        namespace,
    )
    cost = namespace["Cost"]
    arm = np.array(
        [[0, 0], [0.03, 0.08], [0.12, 0.14], [0.24, 0.09], [0.31, -0.12], [0.12, -0.40]]
    )
    fixture = {
        "source_repository": "https://github.com/imphys/MRF_undersampling_optimization",
        "source_revision": REVISION,
        "source_blob": BLOB,
        "source_license": "GPL-3.0",
        "description": "Synthetic supplied spiral; actual unmodified reference helper outputs. No EPG.",
        "base_arm_yx_cycles_per_pixel": arm.tolist(),
    }
    deviations = {}
    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory() as temporary:
        np.savez(Path(temporary) / "Single spiral.npz", Coords=arm[:, ::-1].T)
        try:
            os.chdir(temporary)
            for schedule in ("golden", "reference_arms"):
                reference = []
                for frame in range(2):
                    coords_xy_rad, mask, arm_length = cost.Spiral_coord(
                        frame,
                        "2-pi",
                        (5, 5),
                        schedule == "golden",
                        "Philips_spiral",
                        0.17,
                        3,
                    )
                    if not np.all(mask):
                        raise AssertionError(
                            "Synthetic compatibility case must not be clipped"
                        )
                    reference.append(
                        (coords_xy_rad[:, ::-1] / (2 * np.pi)).reshape(3, arm_length, 2)
                    )
                reference = np.array(reference)
                fixture[schedule] = reference.tolist()
                actual = generate_rotated_spiral_trajectory(
                    arm,
                    2,
                    interleaves_per_frame=3,
                    schedule=schedule,
                    initial_angle_rad=0.17,
                )
                deviations[schedule + "_max_abs"] = float(
                    np.max(np.abs(actual - reference))
                )
        finally:
            os.chdir(original_directory)
    weights_column = cost.Spiral_dcf(arm[:, ::-1], True)
    weights = weights_column[:, 0]
    fixture["radial_weights"] = weights.tolist()
    deviations["dcf_max_abs"] = float(
        np.max(np.abs(weights - radial_increment_density_compensation(arm)))
    )
    reference_psf = cost.P_single(
        cost.Coord((5, 5)), (5, 5), arm[:, ::-1] * 2 * np.pi, weights_column
    )
    fixture["direct_psf_real"] = reference_psf.real.tolist()
    fixture["direct_psf_imag"] = reference_psf.imag.tolist()
    deviations["direct_psf_max_abs"] = float(
        np.max(np.abs(reference_psf - direct_sampling_psf(arm, (5, 5), weights)))
    )
    try:
        column_psf = cost.P_single_fft((5, 5), arm[:, ::-1] * 5, weights_column)
        deviations["external_column_weights"] = "accepted"
        vector_psf = cost.P_single_fft((5, 5), arm[:, ::-1] * 5, weights)
        deviations["column_vs_vector_weights_psf_max_abs"] = float(
            np.max(np.abs(column_psf - vector_psf))
        )
    except Exception as exc:
        chain = []
        while exc is not None:
            chain.append(type(exc).__name__ + ": " + str(exc))
            exc = exc.__cause__
        deviations["external_column_weights"] = chain
    # Explicit compatibility adjustments: vector weights, xy->yx transpose in
    # the output, and /sqrt(V) to convert A^H w to the ideal normal-operator PSF.
    adjusted = cost.P_single_fft((5, 5), arm[:, ::-1] * 5, weights).T / 5
    deviations["external_adjusted_nufft_vs_direct_psf_max_abs"] = float(
        np.max(np.abs(adjusted - reference_psf))
    )
    op = NufftOperator((5, 5), arm)
    deviations["our_default_nufft_psf_vs_direct_max_abs"] = float(
        np.max(np.abs(calculate_point_spread_function(op, weights) - reference_psf))
    )
    accurate = NufftOperator((5, 5), arm, oversampling=2, kernel_width=6)
    deviations["our_accurate_nufft_psf_vs_direct_max_abs"] = float(
        np.max(
            np.abs(calculate_point_spread_function(accurate, weights) - reference_psf)
        )
    )
    return fixture, {
        "reference_revision": REVISION,
        "sigpy_version": sp.__version__,
        "deviations": deviations,
    }


def write_new(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Capture a NEW candidate fixture; never overwrite frozen values",
    )
    args = parser.parse_args()
    fixture, report = compare(args.reference_directory.resolve())
    if args.fixture:
        write_new(args.fixture, fixture)
    write_new(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
