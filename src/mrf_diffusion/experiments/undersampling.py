"""Historical phantom workflow. See docs/validation.md before interpreting maps."""

from pathlib import Path
import numpy as np
import jax.numpy as jnp
from mrf_diffusion import api as epg
from mrf_diffusion.io.results import save_run
from mrf_diffusion.visualization.plots import plot_train
from mrf_diffusion.visualization.phantom import (
    plot_phantom_map,
    plot_reconstruction_maps,
)
from mrf_diffusion.experiments.settings import parse_settings, required_inputs
from mrf_diffusion.experiments.phantom_settings import PhantomSettings
from mrf_diffusion.diffusion.tensor import make_legacy_tensor_grid
from mrf_diffusion.phantoms.checkerboard import make_checkerboard_tensor_phantom
from mrf_diffusion.reconstruction.dictionary import build_magnitude_dictionary
from mrf_diffusion.reconstruction.matching import (
    match_magnitude_dictionary,
    closest_scale_pair,
    stack_scale_maps,
)


def run(settings):
    angle_path, preparation_angle_path, preparation_phase_path = required_inputs(
        settings
    )
    import sigpy as sp
    import UEEphase_DH as uee
    from tqdm import tqdm

    preparation_flip_angles = np.load(preparation_angle_path)
    preparation_phases = np.load(preparation_phase_path)
    output_dir = settings.output_directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if settings.random_seed is not None:
        np.random.seed(settings.random_seed)
    add_noise = settings.add_noise
    density_compensation = settings.density_compensation
    golden_angle = settings.golden_angle
    spiral = settings.spiral_name
    trajectory_offset = settings.offset
    length_module = settings.length
    flip_angle_train = jnp.array(np.load(angle_path))[:length_module]
    phase_train = epg.generate_piecewise_quadratic_phase(
        flip_angle_train, *settings.phase_parameters
    )
    plot_train(flip_angle_train, output_dir, "FA_array.png", "Flip angle [rad]")
    t1_ms = np.geomspace(settings.t1_minimum, settings.t1_maximum, settings.grid_count)
    t2_ms = np.geomspace(settings.t2_minimum, settings.t2_maximum, settings.grid_count)
    shape_scales = np.linspace(
        settings.scale_minimum, settings.scale_maximum, settings.grid_count
    )
    diffusion_scales = np.linspace(
        settings.scale_minimum, settings.scale_maximum, settings.grid_count
    )
    base_tensor = jnp.asarray(settings.base_tensor)
    print("Create dictionary...")
    idx = length_module
    sample_count = idx * settings.simulation.direction_count
    flip_angles = flip_angle_train[:idx]
    dictionary = build_magnitude_dictionary(
        flip_angles,
        t1_ms,
        t2_ms,
        diffusion_scales,
        shape_scales,
        base_tensor,
        phase_train,
        preparation_flip_angles,
        preparation_phases,
        settings.simulation,
        settings.sequence,
    )
    dictionary_matrix = np.array([dictionary[i] for i in dictionary.keys()])
    signal_norm = np.linalg.norm(dictionary_matrix, 2, axis=1)
    normalized_dictionary = dictionary_matrix / signal_norm[:, None]
    print("Create dictionary done!")
    base = (settings.tile_count, settings.tile_count)
    block_size = settings.tile_size
    rad_frac = settings.mask_fraction
    shape = (block_size * base[0], block_size * base[1])
    checkerboard = np.indices(base).sum(axis=0) % 2
    field = np.repeat(np.repeat(checkerboard, block_size, axis=0), block_size, axis=1)
    print(field)
    t1_ms = settings.white_t1_ms + (settings.gray_t1_ms - settings.white_t1_ms) * field
    t2_ms = settings.white_t2_ms + (settings.gray_t2_ms - settings.white_t2_ms) * field
    rho = uee.Cost.Create_mask(shape[0], rad_frac) * np.ones((shape[0], shape[1]))
    object_mask = rho > 0.5
    plot_phantom_map(
        rho * t2_ms,
        output_dir,
        "Sim Original image.png",
        "Relaxation parameters component of the phantom",
        "T2 [ms]",
        limits=(60, 100),
    )
    diffusion_tensor_map = make_checkerboard_tensor_phantom(
        base_tensor,
        settings.tile_count,
        settings.mask_fraction / 2 * settings.tile_count * settings.tile_size,
        settings.tile_size,
    )
    ny, nx, _, _ = diffusion_tensor_map.shape
    anisotropy_map = np.zeros((ny, nx))
    diffusivity_map = np.zeros((ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            voxel_tensor = diffusion_tensor_map[iy, ix]
            anisotropy_map[iy, ix] = epg.fractional_anisotropy(voxel_tensor)
            diffusivity_map[iy, ix] = epg.legacy_mean_diffusivity(voxel_tensor)
    plot_phantom_map(
        np.nan_to_num(anisotropy_map, nan=0.0),
        output_dir,
        "FA_map.png",
        "Fractional anisotropy component of the phantom",
        "FA",
        cmap="magma",
        limits=(0, 1),
    )
    plot_phantom_map(
        diffusivity_map,
        output_dir,
        "MD_map.png",
        "Legacy tensor trace component of the phantom",
        "trace(D) [mm^2/s]",
        cmap="magma",
    )
    error_T1 = np.zeros([1, shape[0], shape[1]])
    error_T2 = np.zeros([1, shape[0], shape[1]])
    # Historical model uses a fixed tensor here, not the spatial tensor phantom.
    signal_tensor = base_tensor
    white_signal = epg.tensor_signal_magnitude(
        flip_angles,
        phase_train=phase_train,
        t1_ms=settings.white_t1_ms,
        t2_ms=settings.white_t2_ms,
        equilibrium_magnetization=1,
        diffusion_tensor=signal_tensor,
        preparation_flip_angles=preparation_flip_angles,
        preparation_phases=preparation_phases,
        sampling=settings.simulation.sampling,
        sampling_offset=settings.simulation.sampling_offset,
        sampling_rate=settings.simulation.sampling_rate,
        direction_count=settings.simulation.direction_count,
        state_count=settings.simulation.state_count,
        include_inversion=settings.simulation.include_inversion,
        sequence=settings.sequence,
    )
    gray_signal = epg.tensor_signal_magnitude(
        flip_angles,
        phase_train=phase_train,
        t1_ms=settings.gray_t1_ms,
        t2_ms=settings.gray_t2_ms,
        equilibrium_magnetization=1,
        diffusion_tensor=signal_tensor,
        preparation_flip_angles=preparation_flip_angles,
        preparation_phases=preparation_phases,
        sampling=settings.simulation.sampling,
        sampling_offset=settings.simulation.sampling_offset,
        sampling_rate=settings.simulation.sampling_rate,
        direction_count=settings.simulation.direction_count,
        state_count=settings.simulation.state_count,
        include_inversion=settings.simulation.include_inversion,
        sequence=settings.sequence,
    )
    phase = uee.Cost.phase_field(shape=shape, order=settings.phase_field_order)
    image_time_series = np.zeros([sample_count, shape[0], shape[1]], dtype=complex)
    image_time_series = np.tile(field[None, :, :], [sample_count, 1, 1]) * np.tile(
        gray_signal[:, None, None], [1, shape[0], shape[1]]
    )
    image_time_series += np.tile(
        np.abs(field[None, :, :] - 1), [sample_count, 1, 1]
    ) * np.tile(white_signal[:, None, None], [1, shape[0], shape[1]])
    image_time_series = rho * image_time_series * phase
    if add_noise:
        std = settings.noise_standard_deviation
        print("Standard deviation is: ", std)
        noise_field = (
            rho * np.random.normal(0, std, shape)
            + rho * np.random.normal(0, std, shape) * 1j
        )
        image_time_series += noise_field
    k_space_arr = []
    reconstructed_images = np.zeros([sample_count, shape[0], shape[1]], dtype=complex)
    for ii in tqdm(range(sample_count), desc="Undersampling the images"):
        spiral_coord, bool_mask, len_one_spiral = uee.Cost.Spiral_coord(
            ii,
            bounds="full",
            shape=shape,
            golden_angle=golden_angle,
            spiral=spiral,
            off_set=trajectory_offset,
            interleaf=1,
        )
        dcf = uee.Cost.Spiral_dcf(spiral_coord, density_compensation)
        spiral_coord = spiral_coord[bool_mask]
        dcf = dcf[bool_mask]
        nufftlinop = sp.linop.NUFFT(shape, spiral_coord)
        k_space = nufftlinop * image_time_series[ii, :, :]
        k_space_arr.append(k_space)
        image = nufftlinop.H * (k_space[:, None] * dcf)
        reconstructed_images[ii, :, :] = image
    keys = list(dictionary.keys())
    t1_map = np.zeros([shape[0], shape[1]])
    t2_map = np.zeros([shape[0], shape[1]])
    magnetization_map = np.zeros([shape[0], shape[1]])
    phase_map = np.zeros([shape[0], shape[1]])
    recovered_shape_scale = np.zeros([shape[0], shape[1]])
    recovered_diffusion_scale = np.zeros([shape[0], shape[1]])
    print("Start Matching...")
    for ii in tqdm(range(shape[0]), desc="Matching the signals to the dictionary"):
        for iii in range(shape[1]):
            if object_mask[ii, iii]:
                res, M0, inprod_val = match_magnitude_dictionary(
                    normalized_dictionary,
                    reconstructed_images[:, ii, iii][:, None],
                    keys,
                    signal_norm,
                )
                print(res[0])
                t1_map[ii, iii] = res[0][0]
                t2_map[ii, iii] = res[0][1]
                # Historical swap retained: key[2] is diffusion scale, not shape scale.
                recovered_shape_scale[ii, iii] = res[0][2]
                recovered_diffusion_scale[ii, iii] = res[0][3]
                magnetization_map[ii, iii] = M0
                phase_map[ii, iii] = -np.angle(inprod_val)
    combined_map = stack_scale_maps(recovered_shape_scale, recovered_diffusion_scale)
    anisotropy_map = np.zeros([shape[0], shape[1]])
    diffusivity_map = np.zeros([shape[0], shape[1]])
    tensor_grid, param_pairs = make_legacy_tensor_grid(
        base_tensor, diffusion_scales, shape_scales
    )
    for i in range(combined_map.shape[0]):
        for j in range(combined_map.shape[1]):
            index1, index2 = closest_scale_pair(param_pairs, combined_map[i, j])
            matched_tensor = tensor_grid[index1, index2]
            anisotropy_map[i, j] = epg.fractional_anisotropy(matched_tensor)
            diffusivity_map[i, j] = epg.legacy_mean_diffusivity(matched_tensor)
    mask_weight = np.sum(rho)
    t1_denominator = np.copy(t1_ms)
    t1_denominator[t1_denominator == 0] = 1
    relative_t1_error = rho * (t1_map - t1_ms) / t1_denominator
    squared_sum_rel_err = np.sum((100 * relative_t1_error) ** 2)
    error_T1[0, :, :] = rho * (t1_map - t1_ms)
    relative_t1_rms = np.sqrt(1 / mask_weight * squared_sum_rel_err)
    print("RMS relative error for T1: ", relative_t1_rms)
    t2_denominator = np.copy(t2_ms)
    t2_denominator[t2_denominator == 0] = 1
    relative_t2_error = rho * (t2_map - t2_ms) / t2_denominator
    squared_sum_rel_err = np.sum((100 * relative_t2_error) ** 2)
    error_T2[0, :, :] = rho * (t2_map - t2_ms)
    relative_t2_rms = np.sqrt(1 / mask_weight * squared_sum_rel_err)
    print("RMS relative error for T2: ", relative_t2_rms)
    plot_reconstruction_maps(
        output_dir,
        t1_map,
        t2_map,
        anisotropy_map,
        diffusivity_map,
        phase_map,
        relative_t1_error,
        relative_t2_error,
        relative_t1_rms,
        relative_t2_rms,
    )
    save_run(
        output_dir,
        settings,
        {
            "t1_map_ms": t1_map,
            "t2_map_ms": t2_map,
            "legacy_anisotropy_map": anisotropy_map,
            "legacy_tensor_trace_map": diffusivity_map,
            "phase_map_rad": phase_map,
        },
        {
            "relative_t1_rms_percent": float(relative_t1_rms),
            "relative_t2_rms_percent": float(relative_t2_rms),
            "warning": "Unvalidated historical reconstruction; fixed signal tensors and swapped dictionary scales retained",
        },
    )


def main(argv=None):
    settings = parse_settings(
        "Legacy phantom experiment (unvalidated reconstruction)",
        PhantomSettings(),
        argv,
    )
    if settings is not None:
        run(settings)


if __name__ == "__main__":
    main()
