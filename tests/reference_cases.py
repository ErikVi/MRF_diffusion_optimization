"""Compute legacy snapshots. No assertion here claims physical correctness."""

import numpy as np
import mrf_diffusion.api as epg
from tests.cases import (
    ANGLES,
    PHASES,
    PARAMETERS,
    TENSOR,
    PREP,
    equilibrium,
    populated_states,
)


def compute_reference_cases():
    state = populated_states()
    fim = np.asarray(epg.scalar_fisher_information(ANGLES, PHASES, *PARAMETERS))
    scale = np.diag(np.asarray(PARAMETERS))
    normalized_fim = scale @ fim @ scale
    return {
        "simple_epg": np.asarray(
            epg.apply_positive_gradient_shift(
                epg.apply_rf_rotation(equilibrium(), 0.47, -0.21)
            )
        ),
        "relaxation": np.asarray(
            epg.relax_and_shift(state, 1000.0, 80.0, 12.0, gradient_enabled=0)
        ),
        "scalar_diffusion": np.asarray(
            epg.relax_diffuse_scalar_and_shift(
                state, 1000.0, 80.0, 4.0, 0.0008, 0.9, wavevector_step=100
            )
        ),
        "tensor_diffusion": np.asarray(
            epg.relax_diffuse_tensor_and_shift(
                state,
                1000.0,
                80.0,
                4.0,
                TENSOR,
                0.9,
                np.array([1.0, 0.0, 0.0]),
                wavevector_step=100,
            )
        ),
        "short_mrf_scalar": np.asarray(
            epg.simulate_scalar_signal(ANGLES, PHASES, *PARAMETERS)
        ),
        "short_mrf_tensor": np.asarray(
            epg.simulate_tensor_signal(
                ANGLES[:3],
                PHASES[:3],
                1000.0,
                80.0,
                0.9,
                TENSOR,
                PREP,
                PREP,
                direction_count=1,
                state_count=5,
            )
        ),
        "scalar_jacobian": np.asarray(
            epg.scalar_signal_jacobian(ANGLES, PHASES, *PARAMETERS)
        ),
        "scalar_fim": fim,
        "relative_standard_deviation_bounds": np.sqrt(
            np.diag(np.linalg.inv(normalized_fim))
        ),
    }
