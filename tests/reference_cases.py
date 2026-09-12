"""Compute legacy snapshots. No assertion here claims physical correctness."""
import numpy as np
import EPG_blocks_jaxcode as epg
from tests.cases import (ANGLES, PHASES, PARAMETERS, TENSOR, PREP,
                         equilibrium, populated_states)


def compute_reference_cases():
    state = populated_states()
    fim = np.asarray(epg.fisher_information(ANGLES, PHASES, *PARAMETERS))
    scale = np.diag(np.asarray(PARAMETERS))
    normalized_fim = scale@fim@scale
    return {
        "simple_epg": np.asarray(epg.epg_grad(epg.epg_rf(equilibrium(), .47, -.21))),
        "relaxation": np.asarray(epg.epg_relax(state, 1000., 80., 12., Gon=0)),
        "scalar_diffusion": np.asarray(epg.epg_grelax(state, 1000., 80., 4., .0008, .9, kstrength=100)),
        "tensor_diffusion": np.asarray(epg.epg_grelax_generalized(state, 1000., 80., 4., TENSOR, .9,
                                                                 np.array([1., 0., 0.]), kstrength=100)),
        "short_mrf_scalar": np.asarray(epg.output_generator(ANGLES, PHASES, *PARAMETERS)),
        "short_mrf_tensor": np.asarray(epg.output_generator_general(ANGLES[:3], PHASES[:3], 1000., 80., .9,
                                                                    TENSOR, PREP, PREP, gradient_number=1, n_states=5)),
        "scalar_jacobian": np.asarray(epg.jacobian_calculator(ANGLES, PHASES, *PARAMETERS)),
        "scalar_fim": fim,
        "relative_standard_deviation_bounds": np.sqrt(np.diag(np.linalg.inv(normalized_fim))),
    }
