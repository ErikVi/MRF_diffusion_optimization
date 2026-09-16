"""Image reconstruction from a declared spatial sampling operator."""

from mrf_diffusion.encoding._validation import sample_weights


def reconstruct_weighted_adjoint(operator, samples, density_weights=None):
    """Return A^H W samples as a complex (H,W) image, without gain correction.

    operator is NufftOperator; samples and optional nonnegative weights have
    shape (K,). None means identity W. Density compensation is applied exactly
    once here, never implicitly by the forward/adjoint operator. This gridding
    reconstruction is not a least-squares inverse of an undersampled operator.
    """
    # Let the operator enforce sample shape before multiplication can broadcast.
    from mrf_diffusion.encoding.nufft import _complex_array

    samples = _complex_array(samples, (len(operator.coordinates),))
    weights = sample_weights(density_weights, len(operator.coordinates))
    return operator.adjoint(samples * weights.astype(samples.real.dtype))
