"""Controlled sequence comparison with mandatory noiseless reference recovery."""

from dataclasses import asdict, replace
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from mrf_diffusion.experiments.settings import parse_settings
from mrf_diffusion.experiments.forward_settings import ForwardSettings
from mrf_diffusion.experiments.quantitative_settings import QuantitativeSettings
from mrf_diffusion.experiments.forward_undersampling import (
    build_phantom,
    build_sequence,
    build_trajectories,
)
from mrf_diffusion.simulation.image_series import (
    simulate_phantom_image_series,
    signal_frame_table,
)
from mrf_diffusion.reconstruction.tensor_dictionary import (
    estimate_dictionary_size,
    generate_tensor_dictionary,
)
from mrf_diffusion.reconstruction.quantitative import match_complex_dictionary
from mrf_diffusion.reconstruction.metrics import evaluate_parameter_maps
from mrf_diffusion.reconstruction.calibrated import reconstruct_with_impulse_gain
from mrf_diffusion.encoding.series import acquire_image_series
from mrf_diffusion.io.results import save_run


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def truth_and_regions(phantom):
    truth = dict(
        t1_ms=phantom.t1_ms,
        t2_ms=phantom.t2_ms,
        md_mm2_per_s=phantom.mean_diffusivity_map,
        fa=phantom.fractional_anisotropy_map,
        proton_density=phantom.proton_density,
    )
    # Regions are defined for evaluation only, including tensor orientation.
    parameters = np.column_stack(
        [
            v[phantom.support]
            for v in (phantom.t1_ms, phantom.t2_ms, phantom.proton_density)
        ]
        + [phantom.diffusion_tensor[phantom.support].reshape(-1, 9)]
    )
    _, inverse = np.unique(parameters, axis=0, return_inverse=True)
    regions = np.full(phantom.support.shape, -1, int)
    regions[phantom.support] = inverse
    return truth, regions


def load_comparison_sequences(settings):
    """Optimized roles require supplied forward TOMLs; never fabricate trains."""
    baseline = build_sequence(settings)
    sequences = {"baseline": baseline}
    missing = []
    configs = {}
    for name, configured in (
        ("flip_optimized", settings.comparison.flip_optimized_config),
        ("joint_optimized", settings.comparison.joint_optimized_config),
    ):
        if not configured:
            missing.append(name)
            continue
        path = (Path(settings.input_directory) / configured).resolve()
        candidate_settings = parse_settings(
            name, ForwardSettings(), ["--config", str(path)]
        )
        if candidate_settings.sequence.rf_phase_mode != "optimized":
            raise ValueError("Comparison optimized roles require paired archive mode")
        candidate = build_sequence(candidate_settings)
        # This comparison controls scan budget/timing and direction ordering.
        if (
            candidate.settings != baseline.settings
            or candidate_settings.simulation != settings.simulation
        ):
            raise ValueError(
                "Controlled comparison requires identical timing, diffusion settings and simulation options"
            )
        for expected, actual in zip(
            signal_frame_table(baseline, settings.simulation),
            signal_frame_table(candidate, settings.simulation),
        ):
            if not np.array_equal(expected, actual):
                raise ValueError(
                    "Comparison sequences must have identical frame/direction ordering"
                )
        sequences[name] = candidate
        configs[name] = dict(
            configuration=asdict(candidate_settings),
            config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            archive_sha256=hashlib.sha256(
                (
                    Path(candidate_settings.input_directory)
                    / candidate_settings.sequence.optimized_archive
                ).read_bytes()
            ).hexdigest(),
        )
    return sequences, missing, configs


def _encode(settings, images, coordinates, noise, seed):
    return acquire_image_series(
        images,
        coordinates,
        oversampling=settings.trajectory.oversampling,
        kernel_width=settings.trajectory.kernel_width,
        noise_std_per_channel=noise,
        noise_seed=seed,
    )


def _match(settings, images, dictionary, support):
    return match_complex_dictionary(
        images,
        dictionary,
        support,
        batch_size=settings.dictionary.voxel_batch_size,
        ambiguity_tolerance=settings.comparison.ambiguity_tolerance,
    )


def reference_recovery_gate(settings, phantom, series, dictionary, coordinates):
    """Validate all dictionary atoms, direct phantom and Cartesian noiseless images.

    A gate failure forbids undersampling interpretation. Discrete tissue recovery
    must be exact within roundoff; density/phase and images allow NUFFT tolerance.
    """
    support = phantom.support
    atoms = _match(
        settings,
        dictionary.signals.T[:, None, :],
        dictionary,
        np.ones((1, len(dictionary.signals)), bool),
    )
    atom_pass = np.array_equal(
        atoms.dictionary_index.ravel(), np.arange(len(dictionary.signals))
    ) and not np.any(atoms.ambiguous)
    encoded = _encode(settings, series.images, coordinates, 0.0, 0)
    frames, _ = reconstruct_with_impulse_gain(encoded)
    fitted = _match(settings, frames, dictionary, support)
    direct = _match(settings, series.images, dictionary, support)
    truth, _ = truth_and_regions(phantom)
    image_error = float(
        np.linalg.norm(frames - series.images) / np.linalg.norm(series.images)
    )
    checks = dict(
        dictionary_self_match=bool(atom_pass),
        direct_valid=bool(
            np.all(direct.valid[support]) and not np.any(direct.ambiguous[support])
        ),
        reconstructed_valid=bool(
            np.all(fitted.valid[support]) and not np.any(fitted.ambiguous[support])
        ),
        image_consistency=image_error
        <= settings.comparison.reference_image_relative_tolerance,
    )
    for name, actual in truth.items():
        tolerance = (
            settings.comparison.reference_density_relative_tolerance
            if name == "proton_density"
            else 1e-9
        )
        checks["direct_" + name] = bool(
            np.allclose(
                direct.maps[name][support], actual[support], rtol=1e-9, atol=1e-12
            )
        )
        checks["recovered_" + name] = bool(
            np.allclose(
                fitted.maps[name][support], actual[support], rtol=tolerance, atol=1e-12
            )
        )
    phase_error = np.angle(
        np.exp(
            1j
            * (
                fitted.maps["object_phase_rad"][support]
                - phantom.object_phase_map[support]
            )
        )
    )
    checks["object_phase"] = bool(
        np.all(
            np.abs(phase_error)
            <= settings.comparison.reference_density_relative_tolerance
        )
    )
    report = dict(
        passed=bool(all(checks.values())),
        checks=checks,
        image_relative_error=image_error,
        minimum_dictionary_correlation_margin=float(np.min(atoms.correlation_margin)),
        maximum_reference_phase_error_rad=float(np.max(np.abs(phase_error))),
    )
    return report


def _save_estimate(
    directory, settings, truth, regions, fitted, frames, acquisition, gains, summary
):
    arrays = {
        **{"truth_" + k: v for k, v in truth.items()},
        **{"recovered_" + k: v for k, v in fitted.maps.items()},
        **{"error_" + k: fitted.maps[k] - v for k, v in truth.items()},
        "recovered_tensor": fitted.tensors,
        "complex_scale": fitted.complex_scale,
        "dictionary_index": fitted.dictionary_index,
        "valid": fitted.valid,
        "ambiguous": fitted.ambiguous,
        "correlation_margin": fitted.correlation_margin,
        "relative_residual": fitted.relative_residual,
        "regions": regions,
        "reconstructed_frames": frames,
        "kspace": acquisition.kspace,
        "coordinates": acquisition.coordinates,
        "reconstruction_gain_divisor": gains,
    }
    save_run(directory, settings, arrays, summary)


def run(settings):
    from mrf_diffusion.visualization.quantitative import (
        plot_parameter_triptychs,
        plot_sampling_comparison,
    )

    c = settings.comparison
    if not c.interleaf_counts or any(
        type(n) is not int or n < 1 for n in c.interleaf_counts
    ):
        raise ValueError("interleaf_counts must be nonempty positive integers")
    if not c.noise_std_per_channel or any(
        not np.isfinite(n) or n < 0 for n in c.noise_std_per_channel
    ):
        raise ValueError("Noise levels must be finite and nonnegative")
    if not c.noise_seeds or any(type(n) is not int or n < 0 for n in c.noise_seeds):
        raise ValueError("Provide nonnegative integer noise seeds")
    for tolerance in (
        c.reference_image_relative_tolerance,
        c.reference_density_relative_tolerance,
        c.ambiguity_tolerance,
    ):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Gate tolerances must be finite and nonnegative")
    if settings.trajectory.kind not in ("generated_spiral", "supplied_spiral"):
        raise ValueError(
            "Comparison spiral conditions require a spiral trajectory configuration"
        )
    sequences, missing, source_configs = load_comparison_sequences(settings)
    phantom = build_phantom(settings)
    if np.any(phantom.proton_density[phantom.support] <= 0):
        raise ValueError(
            "Quantitative validation support must have nonzero proton density"
        )
    truth, regions = truth_and_regions(phantom)
    output = Path(settings.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "status.json",
        {"status": "validating_reference", "missing_sequences": missing},
    )
    np.savez(
        output / "phantom.npz",
        **truth,
        support=phantom.support,
        regions=regions,
        tensor_mm2_per_s=phantom.diffusion_tensor,
        object_phase_map_rad=phantom.object_phase_map,
    )
    frame_count = len(signal_frame_table(sequences["baseline"], settings.simulation)[0])
    estimate = estimate_dictionary_size(settings.dictionary, frame_count)
    _write_json(output / "dictionary_budget.json", estimate)
    cartesian_settings = replace(
        settings,
        trajectory=replace(
            settings.trajectory, kind="cartesian", interleaves_per_frame=1
        ),
    )
    reference = build_trajectories(cartesian_settings, frame_count)
    prepared, gates = {}, {}
    # Validate every supplied sequence before acquiring any undersampled data.
    for name, sequence in sequences.items():
        dictionary = generate_tensor_dictionary(
            settings.dictionary, sequence, settings.simulation
        )
        series = simulate_phantom_image_series(phantom, sequence, settings.simulation)
        gate = reference_recovery_gate(settings, phantom, series, dictionary, reference)
        gates[name] = gate
        _write_json(output / "validation_gates.json", gates)
        if not gate["passed"]:
            _write_json(
                output / "status.json",
                dict(status="reference_recovery_failed", sequence=name, gate=gate),
            )
            raise ValueError(
                f"{name}: fully sampled recovery failed; inspect validation_gates.json. No undersampling interpretation allowed."
            )
        directory = output / name
        directory.mkdir(exist_ok=True)
        archive = dict(
            flip_angles_rad=sequence.flip_angles, rf_phases_rad=sequence.rf_phases
        )
        if sequence.preparation_flip_angles is not None:
            archive.update(
                preparation_flip_angles_rad=sequence.preparation_flip_angles,
                preparation_rf_phases_rad=sequence.preparation_phases,
            )
        np.savez(directory / "sequence.npz", **archive)
        np.savez(
            directory / "signal_frames.npz",
            images=series.images,
            direction_index=series.direction_index,
            readout_index=series.readout_index,
            block=series.block,
            source_sample_index=series.source_sample_index,
        )
        np.savez(
            directory / "dictionary.npz",
            signals=dictionary.signals,
            t1_ms=dictionary.t1_ms,
            t2_ms=dictionary.t2_ms,
            tensors=dictionary.tensors,
        )
        _write_json(
            directory / "sequence_metadata.json",
            dict(
                physics=asdict(sequence.settings),
                simulation=asdict(settings.simulation),
                provenance=source_configs.get(
                    name, {"source": "explicit unoptimized baseline configuration"}
                ),
                sequence_sha256=hashlib.sha256(
                    (directory / "sequence.npz").read_bytes()
                ).hexdigest(),
            ),
        )
        prepared[name] = dictionary, series
    records = []
    conditions = [("cartesian", 1, reference)]
    for arms in c.interleaf_counts:
        configured = replace(
            settings,
            trajectory=replace(settings.trajectory, interleaves_per_frame=arms),
        )
        conditions.append(("spiral", arms, build_trajectories(configured, frame_count)))
    levels = sorted(set((0.0, *c.noise_std_per_channel)))
    for sampling, arms, coordinates in conditions:
        for noise in levels:
            seeds = c.noise_seeds[:1] if noise == 0 else c.noise_seeds
            for seed in seeds:
                # Same coordinates, shape, seed and SD across sequences => same noise array.
                for name, (dictionary, series) in prepared.items():
                    acquisition = _encode(
                        settings, series.images, coordinates, noise, seed
                    )
                    dcf = "none" if sampling == "cartesian" else c.density_compensation
                    frames, gains = reconstruct_with_impulse_gain(
                        acquisition,
                        density_compensation=dcf,
                        pipe_menon_iterations=settings.density_compensation.pipe_menon_iterations,
                        unit_impulse_gain=(
                            c.unit_impulse_gain if sampling == "spiral" else False
                        ),
                    )
                    fitted = _match(settings, frames, dictionary, phantom.support)
                    metrics = evaluate_parameter_maps(
                        truth, fitted.maps, phantom.support, regions
                    )
                    condition = f"{sampling}_L{arms}_noise{noise:g}_seed{seed}"
                    row = dict(
                        sequence=name,
                        condition=condition,
                        sampling=sampling,
                        interleaves=arms,
                        samples_per_frame=int(np.prod(coordinates.shape[1:3])),
                        sample_count_ratio=float(
                            np.prod(phantom.support.shape)
                            / np.prod(coordinates.shape[1:3])
                        ),
                        noise_std_per_channel=noise,
                        seed=seed,
                        density_compensation=dcf,
                        unit_impulse_gain=bool(
                            c.unit_impulse_gain and sampling == "spiral"
                        ),
                        ambiguous_voxels=int(fitted.ambiguous[phantom.support].sum()),
                        metrics=metrics,
                    )
                    records.append(row)
                    destination = output / name / condition
                    _save_estimate(
                        destination,
                        settings,
                        truth,
                        regions,
                        fitted,
                        frames,
                        acquisition,
                        gains,
                        row,
                    )
                    plot_parameter_triptychs(
                        destination,
                        truth,
                        fitted.maps,
                        phantom.support,
                        f"{name}: {sampling}, L={arms}, noise={noise:g}",
                    )
    _write_json(output / "metrics.json", records)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "sequence",
            "condition",
            "sampling",
            "interleaves",
            "samples_per_frame",
            "noise_std_per_channel",
            "seed",
            "region",
            "parameter",
            "mae",
            "rmse",
            "nrmse",
            "bias",
            "valid_voxels",
            "invalid_voxels",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            for region, entries in row["metrics"].items():
                for parameter, metric in entries.items():
                    writer.writerow(
                        {
                            **{key: row[key] for key in fields[:7]},
                            "region": region,
                            "parameter": parameter,
                            **{key: metric[key] for key in fields[9:]},
                        }
                    )
    comparisons = []
    for row in records:
        if row["sequence"] == "baseline":
            continue
        baseline = next(
            r
            for r in records
            if r["sequence"] == "baseline" and r["condition"] == row["condition"]
        )
        differences = {}
        for parameter in truth:
            current = row["metrics"]["all"][parameter]
            base = baseline["metrics"]["all"][parameter]
            comparable = (
                current["invalid_voxels"] == 0
                and base["invalid_voxels"] == 0
                and row["ambiguous_voxels"] == 0
                and baseline["ambiguous_voxels"] == 0
            )
            delta = current["rmse"] - base["rmse"] if comparable else None
            differences[parameter] = dict(
                rmse_difference_vs_baseline=delta,
                result=(
                    "unresolved"
                    if delta is None
                    else (
                        "lower_rmse"
                        if delta < -1e-12
                        else "higher_rmse" if delta > 1e-12 else "equal_within_1e-12"
                    )
                ),
            )
        comparisons.append(
            dict(
                sequence=row["sequence"],
                condition=row["condition"],
                parameters=differences,
            )
        )
    status = dict(
        status="completed_model_consistency_experiment",
        missing_sequences=missing,
        comparisons=comparisons,
        interpretation="Restricted on-grid prolate-tensor experiment using known-defective legacy tensor physics; no general superiority claim.",
        optimized_conclusion=(
            "Unavailable without supplied paired archives"
            if "joint_optimized" in missing
            else "See per-condition comparisons; conditional on model and grid."
        ),
    )
    _write_json(output / "status.json", status)
    plot_sampling_comparison(output, records)
    return records, gates


def main(argv=None):
    settings = parse_settings(
        "Quantitative diffusion-MRF undersampling comparison",
        QuantitativeSettings(),
        argv,
    )
    if settings is not None:
        run(settings)


if __name__ == "__main__":
    main()
