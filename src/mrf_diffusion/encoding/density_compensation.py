"""Explicit alternative density compensation methods, with no gain normalization."""

import numpy as np
from ._validation import coordinates_2d, image_shape_2d, positive_integer, sigpy_module
from .trajectory import to_sigpy_coordinates


def radial_increment_density_compensation(coordinates):
    """Return r*abs(delta r) weights shaped (...,samples) for (...,samples,2).

    First weight of EACH interleaf is zero. Pass arms on a leading axis, not
    concatenated along samples. This is David's explicit radial-increment
    heuristic, useful for his outward spiral; it is NOT universal quadrature.
    Coordinates normally have cycles/pixel units; weights then have their square
    units. Scaling coordinates by c scales weights by c^2. Do not renormalize
    without an explicit amplitude convention. No clipping is performed.
    """
    coordinates = coordinates_2d(coordinates)
    radius = np.linalg.norm(coordinates, axis=-1)
    increments = np.diff(radius, axis=-1, prepend=radius[..., :1])
    return radius * np.abs(increments)


def pipe_menon_density_compensation(
    coordinates,
    image_shape,
    *,
    iterations=30,
):
    """Return (samples,) weights from SigPy's separate Pipe-Menon estimator.

    Input (samples,2) is (ky,kx), cycles/pixel. Uses the supplied image grid and
    upstream default gridding kernel, CPU execution, no progress bar or rescaling.
    These weights are NOT equal in scale or definition to radial-increment DCF.
    """
    coordinates = coordinates_2d(coordinates, single=True)
    shape = image_shape_2d(image_shape)
    iterations = positive_integer(iterations, "iterations")
    sigpy_module()
    from sigpy.mri import pipe_menon_dcf

    weights = np.asarray(
        pipe_menon_dcf(
            to_sigpy_coordinates(coordinates, shape),
            img_shape=shape,
            max_iter=iterations,
            show_pbar=False,
        )
    )
    if (
        weights.shape != (len(coordinates),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
    ):
        raise ValueError("Pipe-Menon returned invalid density weights")
    return weights
