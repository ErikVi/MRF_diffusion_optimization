"""Experiment output management and source provenance; no scientific operations."""

from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import zipfile
import xml.etree.ElementTree as ET
import jax

STAGES = (
    "00_audit",
    "01_initial_sequence",
    "02_flip_angle_optimization",
    "03_phase_method_comparison",
    "04_final_sequence",
    "05_information_validation",
    "06_signal_validation",
    "07_phantom",
    "08_acquisition",
    "09_reconstruction",
    "10_parameter_maps",
    "11_final_comparison",
    "data",
    "report",
)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def initialize_output(config_path, config):
    root = (
        config_path.parent / config["output_directory"] / config["run_name"]
    ).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in STAGES:
        (root / name).mkdir()
    (root / "data/config.toml").write_bytes(config_path.read_bytes())
    evidence = (config_path.parent / config["test_report"]).resolve()
    test_xml = ET.parse(evidence)
    cases = test_xml.findall(".//testcase")
    failures = sum(
        c.find("failure") is not None or c.find("error") is not None for c in cases
    )
    skipped = sum(c.find("skipped") is not None for c in cases)
    if not cases or failures:
        raise ValueError(
            "A passing complete-suite JUnit report is required before this run"
        )
    (root / "00_audit/test_suite.xml").write_bytes(evidence.read_bytes())
    write_json(
        root / "00_audit/test_suite.json",
        dict(
            passed=len(cases) - failures - skipped,
            failures=failures,
            skipped_or_expected_failures=skipped,
            source=str(evidence),
        ),
    )
    (root / "00_audit/AUDIT.md").write_bytes(
        (config_path.parent / "AUDIT.md").read_bytes()
    )
    repo = Path(__file__).resolve().parents[2]
    metadata = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        jax_x64=bool(jax.config.x64_enabled),
        versions={
            n: importlib.metadata.version(n)
            for n in ("jax", "jaxlib", "numpy", "scipy", "sigpy", "matplotlib")
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
        root / "data/source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for folder in ("src", "experiments/complete_worked_example", "tests", "docs"):
            for path in (repo / folder).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(repo))
        for name in (
            "AGENTS.md",
            "pyproject.toml",
            "README.md",
            "requirements-acquisition-validation.txt",
        ):
            archive.write(repo / name, name)
    write_json(root / "data/metadata.json", metadata)
    return root, metadata
