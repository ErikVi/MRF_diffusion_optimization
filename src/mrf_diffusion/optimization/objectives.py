"""optimization / objectives. Numerical behavior retained from the validated baseline."""

from mrf_diffusion.sequence.parameterization import decode_sequence_parameters
from mrf_diffusion.information.crlb import inverse_fisher_information
from mrf_diffusion.sequence.definition import LEGACY_SEQUENCE
import jax
import jax.numpy as jnp
from jax import lax, jacobian
from functools import partial
from mrf_diffusion.diffusion.metrics import fractional_anisotropy
from mrf_diffusion.diffusion.metrics import legacy_mean_diffusivity
from mrf_diffusion.information.fisher import legacy_tensor_fisher_information
from mrf_diffusion.information.fisher import scalar_fisher_information
from mrf_diffusion.sequence.bspline import evaluate_bspline


@partial(jax.jit, static_argnums=(7,))
def scalar_ensemble_objective(
    coefficients,
    phase_train,
    eval_points,
    knots,
    tissues,
    parameter_weights,
    tissue_weights,
    sequence=LEGACY_SEQUENCE,
):
    """Sum tissue-weighted sqrt(sum(parameter_weight*diag(FIM^-1)/theta^2)).

    tissues rows are [T1,T2,D,M]; coefficients reconstruct the flip-angle
    train with a cubic spline. No rank guard or regularization is applied."""
    flip_angle_train = evaluate_bspline(eval_points, knots, coefficients)

    def compute_crlb(params, weight):
        t1_ms, t2_ms, diffusivity, equilibrium_magnetization = params
        FIM = scalar_fisher_information(
            flip_angle_train,
            phase_train,
            t1_ms,
            t2_ms,
            diffusivity,
            equilibrium_magnetization,
            sequence=sequence,
        )
        crlb = jnp.sqrt(
            jnp.sum(
                parameter_weights
                * jnp.diag(inverse_fisher_information(FIM))
                / params**2
            )
        )
        return crlb * weight

    total_crlb = jax.vmap(compute_crlb)(tissues, tissue_weights)
    return jnp.sum(total_crlb)


@partial(jax.jit, static_argnums=(11, 12, 13, 14, 15, 16, 17, 18, 19))
def tensor_ensemble_objective(
    coefficients,
    eval_points,
    knots,
    scalar_tissues,
    diffusion_tensors,
    parameter_weights,
    tissue_weights,
    phase_offset,
    phase_slope,
    preparation_flip_angles,
    preparation_phases,
    sampling=False,
    sampling_offset=0,
    sampling_rate=32,
    direction_count=3,
    state_count=20,
    include_inversion=False,
    aggregation="L2",
    method="quadratic",
    sequence=LEGACY_SEQUENCE,
):
    """Decode a sequence and aggregate relative T1/T2/M precision costs.

    scalar_tissues is (tissues,3), diffusion_tensors (tissues,3,3).
    L1 sums weighted relative SD bounds; L2 sums their squares (no final sqrt).
    Only the first three parameter weights and leading FIM block are used;
    FA/MD are excluded before inversion (I02). Defaults preserve the MSc mode."""
    flip_angle_train, phase_train = decode_sequence_parameters(
        coefficients, eval_points, knots, phase_offset, phase_slope, method
    )
    if aggregation == "L2":
        second_order = 2
    elif aggregation == "L1":
        second_order = 1
    else:
        raise ValueError("Unknown objective aggregation; expected: 'L2' or 'L1' ")
    tissue_anisotropy = jnp.array([fractional_anisotropy(x) for x in diffusion_tensors])
    tissue_legacy_diffusivity = jnp.array(
        [legacy_mean_diffusivity(x) for x in diffusion_tensors]
    )
    anisotropy_column = tissue_anisotropy[:, None]
    legacy_diffusivity_column = tissue_legacy_diffusivity[:, None]
    params_combined = jnp.concatenate(
        [scalar_tissues, anisotropy_column, legacy_diffusivity_column], axis=1
    )
    scalar_tissues = jnp.array(scalar_tissues)
    diffusion_tensors = jnp.array(diffusion_tensors)

    def scan_body(total_crlb_accum, i):
        i = i.astype(int)
        params = params_combined[i]
        t1_ms, t2_ms, equilibrium_magnetization = scalar_tissues[i]
        diffusion_tensor = diffusion_tensors[i]
        FIM = legacy_tensor_fisher_information(
            flip_angle_train,
            phase_train,
            t1_ms,
            t2_ms,
            equilibrium_magnetization,
            diffusion_tensor,
            preparation_flip_angles,
            preparation_phases,
            sampling,
            sampling_offset,
            sampling_rate,
            direction_count,
            state_count,
            include_inversion,
            sequence=sequence,
        )
        crlb = jnp.sum(
            parameter_weights[:3]
            * jnp.sqrt(
                jnp.diag(inverse_fisher_information(FIM[:-2, :-2])) / params[:3] ** 2
            )
            ** second_order
        )
        weighted_crlb = crlb * tissue_weights[i]
        return (total_crlb_accum + weighted_crlb, None)

    total_crlb, _ = lax.scan(scan_body, 0.0, jnp.arange(len(params_combined)))
    return total_crlb
