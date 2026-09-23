"""Experiment-specific assembly of existing phantom, encoding and fitting APIs."""

from dataclasses import replace
import numpy as np
from mrf_diffusion.experiments.forward_undersampling import (
    build_phantom,
    build_trajectories,
)
from mrf_diffusion.experiments.quantitative_undersampling import (
    truth_and_regions,
    reference_recovery_gate,
)
from mrf_diffusion.simulation.image_series import simulate_phantom_image_series
from mrf_diffusion.simulation.api import simulate_mrf_signal
from mrf_diffusion.simulation.tissue import TissueParameters
from mrf_diffusion.reconstruction.tensor_dictionary import (
    generate_tensor_dictionary,
    estimate_dictionary_size,
)
from mrf_diffusion.encoding.series import acquire_image_series
from mrf_diffusion.reconstruction.calibrated import reconstruct_with_impulse_gain
from mrf_diffusion.reconstruction.quantitative import match_complex_dictionary
from mrf_diffusion.reconstruction.metrics import evaluate_parameter_maps
from mrf_diffusion.diffusion.parameterization import tensor_invariants
from output import write_json, write_csv


def run_spatial_validation(root, config, settings, optimization, sequences):
    phantom = build_phantom(settings)
    truth, regions = truth_and_regions(phantom)
    measured_md, measured_fa, _ = tensor_invariants(phantom.diffusion_tensor)
    # Check against requested compartment values, not merely the cached invariants.
    for i, t1 in enumerate(settings.phantom.t1_ms):
        mask = phantom.support & (phantom.t1_ms == t1)
        np.testing.assert_allclose(
            measured_md[mask],
            settings.diffusion.mean_diffusivity_mm2_per_s[i],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            measured_fa[mask], settings.diffusion.fractional_anisotropy[i], atol=1e-14
        )
    np.savez(
        root / "07_phantom/ground_truth.npz",
        **truth,
        support=phantom.support,
        regions=regions,
        diffusion_tensor=phantom.diffusion_tensor,
        object_phase_map=phantom.object_phase_map,
    )
    options = optimization.simulation
    prepared, gates, checks = {}, {}, {}
    for name, sequence in sequences.items():
        print("Reference simulation", name, flush=True)
        series = simulate_phantom_image_series(phantom, sequence, options)
        write_json(
            root / "data/dictionary_budget.json",
            estimate_dictionary_size(settings.dictionary, len(series.images)),
        )
        dictionary = generate_tensor_dictionary(settings.dictionary, sequence, options)
        reference = build_trajectories(
            replace(
                settings,
                trajectory=replace(
                    settings.trajectory, kind="cartesian", interleaves_per_frame=1
                ),
            ),
            len(series.images),
        )
        gates[name] = reference_recovery_gate(
            settings, phantom, series, dictionary, reference
        )
        write_json(root / "09_reconstruction/reference_gates.json", gates)
        np.savez(
            root / "06_signal_validation" / (name + "_spatial.npz"),
            images=series.images,
            dictionary=dictionary.signals,
            t1_ms=dictionary.t1_ms,
            t2_ms=dictionary.t2_ms,
            tensors=dictionary.tensors,
            direction_index=series.direction_index,
            readout_index=series.readout_index,
            block=series.block,
            source_sample_index=series.source_sample_index,
        )
        if not gates[name]["passed"]:
            raise ValueError(
                "Reference recovery failed; no undersampling interpretation"
            )
        state_errors = []
        for i in (0, len(dictionary.signals) - 1):
            tissue = TissueParameters(
                float(dictionary.t1_ms[i]),
                float(dictionary.t2_ms[i]),
                1.0,
                dictionary.tensors[i],
            )
            raw = np.asarray(
                simulate_mrf_signal(
                    tissue,
                    sequence,
                    replace(options, state_count=2 * options.state_count),
                )
            )
            larger = raw[0] + 1j * raw[1]
            error = float(
                np.linalg.norm(larger - dictionary.signals[i]) / np.linalg.norm(larger)
            )
            state_errors.append(error)
            np.testing.assert_allclose(
                larger, dictionary.signals[i], atol=1e-12, rtol=1e-12
            )
        checks[name] = dict(
            doubled_state_relative_errors=state_errors,
            unique_tissues=series.unique_signal_count,
        )
        prepared[name] = (series, dictionary)
    write_json(root / "06_signal_validation/state_convergence.json", checks)
    # All sequence reference gates have passed before any spiral condition.
    conditions = [("cartesian", 1, reference)]
    for arms in config["interleaf_counts"]:
        coords = build_trajectories(
            replace(
                settings,
                trajectory=replace(settings.trajectory, interleaves_per_frame=arms),
            ),
            len(reference),
        )
        conditions.append(("spiral", arms, coords))
    records, flat, noise_checks = [], [], []
    for noise in config["noise_levels"]:
        for sampling, arms, coords in conditions:
            condition = f"{sampling}_L{arms}_noise{noise:g}"
            np.save(
                root / "08_acquisition" / (f"{sampling}_L{arms}_coordinates.npy"),
                coords,
            )
            paired_noise = None
            for name, (series, dictionary) in prepared.items():
                print("Acquire/match", name, condition, flush=True)
                acquisition = acquire_image_series(
                    series.images,
                    coords,
                    oversampling=settings.trajectory.oversampling,
                    kernel_width=settings.trajectory.kernel_width,
                    noise_std_per_channel=noise,
                    noise_seed=config["noise_seed"],
                )
                key = name + "_" + condition
                if noise:
                    clean_key = name + f"_{sampling}_L{arms}_noise0"
                    clean = np.load(root / "08_acquisition" / (clean_key + ".npz"))[
                        "kspace"
                    ]
                    realization = acquisition.kspace - clean
                    if paired_noise is None:
                        paired_noise = realization
                    noise_error = float(np.max(np.abs(realization - paired_noise)))
                    np.testing.assert_allclose(
                        realization, paired_noise, atol=1e-14, rtol=0
                    )
                    noise_checks.append(
                        dict(
                            sequence=name,
                            condition=condition,
                            paired_noise_max_difference=noise_error,
                            rms_signal_over_rms_noise=float(
                                np.linalg.norm(clean) / np.linalg.norm(realization)
                            ),
                        )
                    )
                frames, gains = reconstruct_with_impulse_gain(
                    acquisition,
                    density_compensation=(
                        "none"
                        if sampling == "cartesian"
                        else settings.comparison.density_compensation
                    ),
                    unit_impulse_gain=sampling == "spiral"
                    and settings.comparison.unit_impulse_gain,
                )
                fitted = match_complex_dictionary(
                    frames,
                    dictionary,
                    phantom.support,
                    batch_size=settings.dictionary.voxel_batch_size,
                    ambiguity_tolerance=settings.comparison.ambiguity_tolerance,
                )
                metrics = evaluate_parameter_maps(
                    truth, fitted.maps, phantom.support, regions
                )
                np.savez(
                    root / "08_acquisition" / (key + ".npz"), kspace=acquisition.kspace
                )
                np.savez(
                    root / "09_reconstruction" / (key + ".npz"),
                    frames=frames,
                    gains=gains,
                )
                np.savez(
                    root / "10_parameter_maps" / (key + ".npz"),
                    **fitted.maps,
                    **{"error_" + k: fitted.maps[k] - v for k, v in truth.items()},
                    recovered_tensor=fitted.tensors,
                    valid=fitted.valid,
                    ambiguous=fitted.ambiguous,
                    dictionary_index=fitted.dictionary_index,
                    complex_scale=fitted.complex_scale,
                    correlation_margin=fitted.correlation_margin,
                    relative_residual=fitted.relative_residual,
                )
                records.append(
                    dict(
                        sequence=name,
                        condition=condition,
                        sampling=sampling,
                        interleaves=arms,
                        noise=noise,
                        ambiguous=int(fitted.ambiguous[phantom.support].sum()),
                        metrics=metrics,
                    )
                )
                for region, parameters in metrics.items():
                    for parameter, metric in parameters.items():
                        flat.append(
                            dict(
                                sequence=name,
                                condition=condition,
                                interleaves=arms,
                                noise=noise,
                                seed=config["noise_seed"],
                                region=region,
                                parameter=parameter,
                                **metric,
                            )
                        )
                write_json(root / "11_final_comparison/results.json", records)
                write_csv(root / "reconstruction_metrics.csv", flat)
    write_json(root / "08_acquisition/noise_checks.json", noise_checks)
