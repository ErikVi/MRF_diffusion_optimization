"""Build a magnitude dictionary on the historical tissue and tensor-scale grid."""

from mrf_diffusion import api as epg
from mrf_diffusion.diffusion.tensor import make_legacy_tensor_grid


def build_magnitude_dictionary(
    flip_angles,
    t1_ms,
    t2_ms,
    diffusion_scales,
    shape_scales,
    base_tensor,
    phase_train,
    preparation_flip_angles,
    preparation_phases,
    simulation,
    sequence,
):
    D_mtrx = make_legacy_tensor_grid(base_tensor, diffusion_scales, shape_scales)[0]
    dictionary = {}
    for i in range(t1_ms.shape[0]):
        for ii in range(t2_ms.shape[0]):
            if t2_ms[ii] < t1_ms[i]:
                for iii in range(diffusion_scales.shape[0]):
                    for iiii in range(shape_scales.shape[0]):
                        # Dictionary construction is a host operation; JAX scalar arrays
                        # are unhashable, unlike the original NumPy grid scalars.
                        key = tuple(
                            float(value)
                            for value in (
                                t1_ms[i],
                                t2_ms[ii],
                                diffusion_scales[iii],
                                shape_scales[iiii],
                            )
                        )
                        dictionary[key] = epg.tensor_signal_magnitude(
                            flip_angles,
                            phase_train=phase_train,
                            t1_ms=t1_ms[i],
                            t2_ms=t2_ms[ii],
                            equilibrium_magnetization=1,
                            diffusion_tensor=D_mtrx[iii][iiii],
                            preparation_flip_angles=preparation_flip_angles,
                            preparation_phases=preparation_phases,
                            sampling=simulation.sampling,
                            sampling_offset=simulation.sampling_offset,
                            sampling_rate=simulation.sampling_rate,
                            direction_count=simulation.direction_count,
                            state_count=simulation.state_count,
                            include_inversion=simulation.include_inversion,
                            sequence=sequence,
                        )
    return dictionary
