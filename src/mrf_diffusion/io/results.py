"""Run provenance and arrays, independent of plotting and optimization."""

from dataclasses import asdict
from pathlib import Path
from importlib.metadata import version
import json
import numpy as np


def save_run(directory, settings, arrays, summary):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(directory / (name + ".npy"), np.asarray(array))
    metadata = {
        "configuration": asdict(settings),
        "versions": {p: version(p) for p in ("jax", "jaxlib", "numpy", "scipy")},
        "result": summary,
    }
    (directory / "run.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
