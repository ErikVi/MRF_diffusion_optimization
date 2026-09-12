"""Explicit manual capture to a NEW path; tests never update their references.

Usage: python tools/capture_validation_reference.py path/to/candidate.json
Review differences and their scientific cause before replacing a committed baseline.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import numpy as np
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
from tests.reference_cases import compute_reference_cases
from tests.cases import ANGLES, PHASES, PARAMETERS, TENSOR


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    destination = Path(sys.argv[1])
    if destination.exists():
        raise SystemExit("Refusing to overwrite an existing reference")
    values = compute_reference_cases()
    result = {
        "classification": "legacy characterization; known defects intentionally retained",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": hashlib.sha256((ROOT/"EPG_blocks_jaxcode.py").read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "backend": jax.default_backend(), "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "versions": {p: importlib.metadata.version(p) for p in ["jax", "jaxlib", "numpy", "scipy", "pytest"]},
        "inputs": {"angles_rad": np.asarray(ANGLES).tolist(), "phases_rad": np.asarray(PHASES).tolist(),
                   "T1_T2_D_M": np.asarray(PARAMETERS).tolist(), "tensor_mm2_per_s": np.asarray(TENSOR).tolist()},
        "cases": {},
    }
    for name, value in values.items():
        # FIM inversion magnifies roundoff; operator references use tighter tolerance.
        rtol = 2e-6 if "bounds" in name else 2e-8
        entry = {"real": value.real.tolist(), "rtol": rtol, "atol": 1e-11}
        if np.iscomplexobj(value):
            entry["imag"] = value.imag.tolist()
        result["cases"][name] = entry
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
