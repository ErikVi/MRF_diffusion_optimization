"""Shared array contracts for spatial encoding (no physics dependencies)."""

from operator import index
import numpy as np


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def image_shape_2d(shape):
    if len(shape) != 2:
        raise ValueError("image_shape must contain exactly (height, width)")
    return tuple(positive_integer(n, "image dimension") for n in shape)


def real_finite(value, name):
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be real numeric values")
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def coordinates_2d(value, *, single=False):
    value = real_finite(value, "coordinates")
    if value.ndim < 2 or value.shape[-1] != 2 or value.shape[-2] < 1:
        raise ValueError("coordinates must have shape (..., samples, 2), samples > 0")
    if single and value.ndim != 2:
        raise ValueError("one acquisition requires coordinates shaped (samples, 2)")
    return value


def sample_weights(value, sample_count):
    if value is None:
        return np.ones(sample_count, dtype=np.float64)
    value = real_finite(value, "density weights")
    if value.shape != (sample_count,) or np.any(value < 0):
        raise ValueError(
            "density weights must have shape (samples,) and be nonnegative"
        )
    return value


def sigpy_module():
    try:
        import sigpy
    except ImportError as exc:
        raise ImportError(
            "Spatial NUFFT requires SigPy; install mrf-diffusion[acquisition]"
        ) from exc
    return sigpy
