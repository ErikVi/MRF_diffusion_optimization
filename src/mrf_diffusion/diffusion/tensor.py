"""diffusion / tensor. Numerical behavior retained from the validated baseline."""

import jax
import jax.numpy as jnp


@jax.jit
def make_scaled_tensor_template(scale):
    """Return a fixed symmetric template scaled in mm^2/s; scale is not MD.

    Diagonals are [1.8,1,.2], off-diagonals .1. This is a historical
    experiment template, not a general tensor parameterization.
    """
    return jnp.array(
        [
            [scale * 1.8, scale * 0.1, scale * 0.1],
            [scale * 0.1, scale * 1.0, scale * 0.1],
            [scale * 0.1, scale * 0.1, scale * 0.2],
        ]
    )


def make_legacy_tensor_grid(base_tensor, diffusion_scales, shape_scales):
    """Return tensors (M,F,3,3) and [diffusion_scale,shape_scale] pairs (M,F,2).

    Scale yy/zz by .1+.9*shape_scale, then multiply the whole tensor by
    diffusion_scale. These coordinates are not physical MD/FA; no PSD check.
    """
    tensor_rows = []
    param_pairs = []
    for diffusion_scale in diffusion_scales:
        row_tensors = []
        row_params = []
        for shape_scale in shape_scales:
            scale_yyzz = 0.1 + 0.9 * shape_scale
            modified_tensor = base_tensor.at[1, 1].set(base_tensor[1, 1] * scale_yyzz)
            modified_tensor = modified_tensor.at[2, 2].set(
                base_tensor[2, 2] * scale_yyzz
            )
            row_tensors.append(diffusion_scale * modified_tensor)
            row_params.append(jnp.array([diffusion_scale, shape_scale]))
        tensor_rows.append(jnp.stack(row_tensors))
        param_pairs.append(jnp.stack(row_params))
    return (jnp.stack(tensor_rows), jnp.stack(param_pairs))
