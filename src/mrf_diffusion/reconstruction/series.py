"""Adjoint frame reconstruction; no quantitative parameter estimation."""

import numpy as np
from mrf_diffusion.encoding.nufft import NufftOperator
from mrf_diffusion.encoding.density_compensation import (
    radial_increment_density_compensation,
    pipe_menon_density_compensation,
)
from .images import reconstruct_weighted_adjoint


def reconstruct_acquisition_frames(
    acquisition, *, density_compensation="none", pipe_menon_iterations=30
):
    """Return complex (F,H,W) frames; DCF calculated per arm before flattening."""
    if density_compensation not in ("none", "radial_increment", "pipe_menon"):
        raise ValueError("Unknown density compensation method")
    frames = []
    for coordinates, samples in zip(acquisition.coordinates, acquisition.kspace):
        flat = coordinates.reshape(-1, 2)
        weights = None
        if density_compensation == "radial_increment":
            weights = radial_increment_density_compensation(coordinates).reshape(-1)
        elif density_compensation == "pipe_menon":
            weights = pipe_menon_density_compensation(
                flat, acquisition.image_shape, iterations=pipe_menon_iterations
            )
        operator = NufftOperator(
            acquisition.image_shape,
            flat,
            acquisition.oversampling,
            acquisition.kernel_width,
        )
        frames.append(
            reconstruct_weighted_adjoint(operator, samples.reshape(-1), weights)
        )
    return np.asarray(frames)
