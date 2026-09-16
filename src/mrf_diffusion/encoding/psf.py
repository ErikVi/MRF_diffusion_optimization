"""Point-spread functions are diagnostics of sampling/reconstruction, not images."""

import numpy as np
from ._validation import coordinates_2d, image_shape_2d, sample_weights


def calculate_point_spread_function(operator, density_weights=None):
    """Return A^H W A delta for a unit central pixel on operator.image_shape.

    No peak/sum normalization is applied. Includes the actual NUFFT
    approximation. This central impulse response need not be exactly shift
    invariant at finite interpolation accuracy.
    """
    weights = sample_weights(density_weights, len(operator.coordinates))
    impulse = np.zeros(operator.image_shape, dtype=np.complex128)
    impulse[tuple(n // 2 for n in operator.image_shape)] = 1
    return operator.adjoint(weights * operator.forward(impulse))


def direct_sampling_psf(
    coordinates, image_shape, density_weights=None, *, displacement_shape=None
):
    """Exact Fourier PSF: sum_k w_k exp(+2*pi*i*k.r)/(H*W).

    coordinates=(K,2), cycles/pixel (ky,kx). image_shape determines normalization.
    displacement_shape defaults to image_shape; choose (2H-1,2W-1) to represent
    every displacement for a linear convolution over an H by W support.
    r=index-floor(displacement_shape/2), in pixels. No source image padding,
    coordinate rescaling or library oversampling is implied. A sample loop
    bounds memory; use this small-case oracle, not a high-throughput gridding API.
    """
    coordinates = coordinates_2d(coordinates, single=True)
    shape = image_shape_2d(image_shape)
    output_shape = (
        shape if displacement_shape is None else image_shape_2d(displacement_shape)
    )
    weights = sample_weights(density_weights, len(coordinates))
    rows, columns = np.meshgrid(
        np.arange(output_shape[0]) - output_shape[0] // 2,
        np.arange(output_shape[1]) - output_shape[1] // 2,
        indexing="ij",
    )
    result = np.zeros(output_shape, dtype=np.complex128)
    for (ky, kx), weight in zip(coordinates, weights):
        result += weight * np.exp(2j * np.pi * (ky * rows + kx * columns))
    return result / np.prod(shape)
