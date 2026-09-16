"""Forward-only diffusion-MRF phantom orchestration. No quantitative matching."""

from pathlib import Path
import hashlib
import numpy as np
from mrf_diffusion.experiments.forward_settings import ForwardSettings
from mrf_diffusion.experiments.settings import parse_settings


def build_phantom(settings):
    from mrf_diffusion.phantoms.geometry import (
        checkerboard_labels,
        circular_support,
        quadratic_object_phase_map,
    )
    from mrf_diffusion.phantoms.maps import assign_tissue_maps

    p, d = settings.phantom, settings.diffusion
    labels = checkerboard_labels(p.image_shape, p.tile_shape)
    if p.geometry == "homogeneous":
        labels = np.zeros_like(labels)
    elif p.geometry != "checkerboard":
        raise ValueError("geometry must be homogeneous or checkerboard")
    support = circular_support(p.image_shape, p.radius_pixels)
    object_phase_map = quadratic_object_phase_map(
        p.image_shape,
        settings.object_phase.edge_phase_rad if settings.object_phase.enabled else 0.0,
    )
    return assign_tissue_maps(
        labels,
        support,
        t1_ms=p.t1_ms,
        t2_ms=p.t2_ms,
        proton_density=p.proton_density,
        mean_diffusivity=d.mean_diffusivity_mm2_per_s,
        fractional_anisotropy=d.fractional_anisotropy,
        principal_directions=d.principal_directions_xyz,
        object_phase_map=object_phase_map,
    )


def build_sequence(settings):
    from mrf_diffusion.sequence.definition import MRFSequence
    from mrf_diffusion.sequence.artifacts import load_sequence_archive
    from mrf_diffusion.sequence.phase import generate_phase_train

    s = settings.sequence
    if s.rf_phase_mode == "optimized":
        if not s.optimized_archive:
            raise ValueError("optimized phase mode requires a paired sequence archive")
        return load_sequence_archive(
            Path(settings.input_directory) / s.optimized_archive, s.physics
        )
    angles = np.asarray(s.flip_angles_rad, float)
    if s.rf_phase_mode == "none":
        phases = np.zeros_like(angles)
    elif s.rf_phase_mode == "explicit":
        phases = np.asarray(s.rf_phases_rad, float)
    elif s.rf_phase_mode == "reference":
        if s.reference_method not in (
            "quadratic",
            "linear",
            "sinusoidal",
            "alternating",
        ):
            raise ValueError("Unknown reference phase family")
        phases = np.asarray(
            generate_phase_train(angles, s.reference_fraction, s.reference_method)
        )
    else:
        raise ValueError("rf_phase_mode must be none, reference, explicit or optimized")
    return MRFSequence(angles, phases, s.physics)


def build_trajectories(settings, frame_count):
    from mrf_diffusion.encoding.trajectory import (
        cartesian_trajectory,
        generate_variable_density_spiral,
        load_spiral_coordinates,
        generate_rotated_spiral_trajectory,
    )

    t = settings.trajectory
    if t.kind == "cartesian":
        if t.interleaves_per_frame != 1:
            raise ValueError("Cartesian reference has one complete grid per frame")
        base = cartesian_trajectory(settings.phantom.image_shape)
        return np.broadcast_to(base, (frame_count, 1, *base.shape)).copy()
    if t.kind == "supplied_spiral":
        base = load_spiral_coordinates(
            Path(settings.input_directory) / t.supplied_archive,
            layout=t.layout,
            component_order=t.component_order,
        )
    elif t.kind == "generated_spiral":
        if tuple(settings.phantom.image_shape) != (
            t.design_matrix_size,
            t.design_matrix_size,
        ):
            raise ValueError(
                "Generated spiral design matrix must match the square image grid"
            )
        base = generate_variable_density_spiral(
            field_of_view_m=t.field_of_view_m,
            matrix_size=t.design_matrix_size,
            frequency_sampling_factor=t.frequency_sampling_factor,
            acceleration=t.acceleration,
            density_exponent=t.density_exponent,
            max_gradient_t_per_m=t.max_gradient_t_per_m,
            max_slew_t_per_m_per_s=t.max_slew_t_per_m_per_s,
        )
    else:
        raise ValueError("Unknown trajectory kind")
    return generate_rotated_spiral_trajectory(
        base,
        frame_count,
        interleaves_per_frame=t.interleaves_per_frame,
        schedule=t.schedule,
        angular_increment_rad=t.angular_increment_rad,
        initial_angle_rad=t.initial_angle_rad,
        reference_arm_count=t.reference_arm_count,
    )


def run(settings):
    from importlib.metadata import version
    from mrf_diffusion.simulation.image_series import simulate_phantom_image_series
    from mrf_diffusion.encoding.series import acquire_image_series
    from mrf_diffusion.reconstruction.series import reconstruct_acquisition_frames
    from mrf_diffusion.io.results import save_run
    from mrf_diffusion.visualization.forward_phantom import plot_forward_experiment

    phantom, sequence = build_phantom(settings), build_sequence(settings)
    series = simulate_phantom_image_series(phantom, sequence, settings.simulation)
    trajectories = build_trajectories(settings, len(series.images))
    acquisition = acquire_image_series(
        series.images,
        trajectories,
        oversampling=settings.trajectory.oversampling,
        kernel_width=settings.trajectory.kernel_width,
        noise_std_per_channel=(
            settings.noise.standard_deviation_per_channel
            if settings.noise.enabled
            else 0.0
        ),
        noise_seed=settings.noise.seed,
    )
    reconstructed = None
    if settings.reconstruction.enabled:
        reconstructed = reconstruct_acquisition_frames(
            acquisition,
            density_compensation=settings.density_compensation.method,
            pipe_menon_iterations=settings.density_compensation.pipe_menon_iterations,
        )
    arrays = {
        "support": phantom.support,
        "ground_truth_t1_ms": phantom.t1_ms,
        "ground_truth_t2_ms": phantom.t2_ms,
        "ground_truth_proton_density": phantom.proton_density,
        "ground_truth_md_mm2_per_s": phantom.mean_diffusivity_map,
        "ground_truth_fa": phantom.fractional_anisotropy_map,
        "ground_truth_tensor_mm2_per_s": phantom.diffusion_tensor,
        "ground_truth_principal_direction_xyz": phantom.principal_direction_map,
        "object_phase_map_rad": phantom.object_phase_map,
        "flip_angles_rad": sequence.flip_angles,
        "rf_phase_train_rad": sequence.rf_phases,
        "complex_image_series": series.images,
        "kspace": acquisition.kspace,
        "trajectory_ky_kx_cycles_per_pixel": acquisition.coordinates,
        "frame_direction_index": series.direction_index,
        "frame_readout_index": series.readout_index,
        "frame_block": series.block,
        "source_sample_index": series.source_sample_index,
    }
    if sequence.preparation_flip_angles is not None:
        arrays["preparation_flip_angles_rad"] = sequence.preparation_flip_angles
        arrays["preparation_rf_phases_rad"] = sequence.preparation_phases
    if reconstructed is not None:
        arrays["adjoint_frames"] = reconstructed
    source_hash = None
    if settings.sequence.rf_phase_mode == "optimized":
        source_hash = hashlib.sha256(
            (
                Path(settings.input_directory) / settings.sequence.optimized_archive
            ).read_bytes()
        ).hexdigest()
    save_run(
        settings.output_directory,
        settings,
        arrays,
        {
            "unique_signal_count": series.unique_signal_count,
            "frame_count": len(series.images),
            "image_order": "frames,y,x",
            "kspace_order": "frames,interleaves,samples",
            "sigpy_version": version("sigpy"),
            "sequence_archive_sha256": source_hash,
            "model_status": "Existing tensor EPG with known D01/D02 defects; forward wiring validation only",
            "density_model": "External proton density multiplier, equilibrium M=1; not quantitative M0 fitting",
        },
    )
    plot_forward_experiment(
        settings.output_directory,
        phantom,
        series,
        acquisition,
        reconstructed,
        settings.reconstruction.representative_frames,
    )
    return phantom, series, acquisition, reconstructed


def main(argv=None):
    settings = parse_settings(
        "Forward diffusion-MRF phantom (no parameter matching)", ForwardSettings(), argv
    )
    if settings is not None:
        run(settings)


if __name__ == "__main__":
    main()
