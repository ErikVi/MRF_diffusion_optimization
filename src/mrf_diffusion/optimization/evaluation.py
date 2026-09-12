"""optimization / evaluation. Numerical behavior retained from the validated baseline."""

from mrf_diffusion.sequence.parameterization import decode_sequence_parameters
from mrf_diffusion.information.crlb import inverse_fisher_information
from mrf_diffusion.sequence.definition import LEGACY_SEQUENCE
import jax
import jax.numpy as jnp
from functools import partial
from mrf_diffusion.diffusion.metrics import fractional_anisotropy
from mrf_diffusion.diffusion.metrics import legacy_mean_diffusivity
from mrf_diffusion.information.fisher import legacy_tensor_fisher_information
from mrf_diffusion.information.fisher import scalar_fisher_information
from mrf_diffusion.sequence.bspline import evaluate_bspline


@partial(jax.jit, static_argnums=(7,))
def scalar_sequence_precision_bounds(
    coefficients,
    phase_train,
    eval_points,
    knots,
    tissues,
    parameter_weights,
    tissue_weights,
    sequence=LEGACY_SEQUENCE,
):
    """Return four weighted relative SD bounds in [T1,T2,D,M] order.

    Known defect I01: every tissue uses tissue_weights[0]. This is retained;
    scalar_ensemble_objective uses each tissue weight correctly."""
    flip_angle_train = evaluate_bspline(eval_points, knots, coefficients)
    crlb_array = jnp.zeros(len(tissues[0]))

    def scan_fn(carry, params):
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
        crlb = parameter_weights * jnp.sqrt(
            jnp.diag(inverse_fisher_information(FIM) / params**2)
        )
        return (carry, crlb * tissue_weights[carry])

    _, crlb_contributions = jax.lax.scan(scan_fn, 0, tissues)
    return jnp.sum(crlb_contributions, axis=0)


@partial(jax.jit, static_argnums=(11, 12, 13, 14, 15, 16, 17, 18))
def tensor_sequence_precision_bounds(
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
    method="quadratic",
    sequence=LEGACY_SEQUENCE,
):
    """Return THREE weighted relative SD bounds [T1,T2,M], not five.

    Decode coefficients using the selected phase method. Invert only the
    leading (3,3) FIM block and normalize by scalar tissue parameters.
    Diffusion is effectively fixed for these bounds; see discrepancy I02."""
    flip_angle_train, phase_train = decode_sequence_parameters(
        coefficients, eval_points, knots, phase_offset, phase_slope, method
    )
    tissue_anisotropy = jnp.array([fractional_anisotropy(x) for x in diffusion_tensors])
    tissue_legacy_diffusivity = jnp.array(
        [legacy_mean_diffusivity(x) for x in diffusion_tensors]
    )
    params_combined = jnp.concatenate(
        [
            scalar_tissues,
            tissue_anisotropy[:, None],
            tissue_legacy_diffusivity[:, None],
        ],
        axis=1,
    )

    def compute_crlb(params, params_scalars, params_difs, weight):
        t1_ms, t2_ms, equilibrium_magnetization = params_scalars
        diffusion_tensor = params_difs
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
        crlb = parameter_weights[:3] * jnp.sqrt(
            jnp.diag(inverse_fisher_information(FIM[:-2, :-2])) / params[:3] ** 2
        )
        return crlb * weight

    crlb_array = jax.vmap(compute_crlb)(
        params_combined, scalar_tissues, diffusion_tensors, tissue_weights
    )
    return jnp.sum(crlb_array, axis=0)
