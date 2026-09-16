"""Forward acquisition of complex image frames; no MRF physics or matching."""

from dataclasses import dataclass
import numpy as np
from .nufft import NufftOperator
from ._validation import coordinates_2d


@dataclass(frozen=True)
class AcquisitionResult:
    """kspace=(frames,interleaves,samples), coordinates=(frames,interleaves,samples,2).

    Coordinates are (ky,kx), cycles/pixel. Noise is independent circular complex
    Gaussian per sample, with noise_std_per_channel for EACH real/imag component.
    Forward k-space is unweighted; DCF belongs to reconstruction, never sampling.
    """

    kspace: np.ndarray
    coordinates: np.ndarray
    image_shape: tuple
    oversampling: float
    kernel_width: float
    noise_std_per_channel: float
    noise_seed: int | None


def acquire_image_series(
    images,
    coordinates,
    *,
    oversampling=1.25,
    kernel_width=4,
    noise_std_per_channel=0.0,
    noise_seed=None,
):
    """Forward NUFFT per frame, preserving interleaves and independent image phase."""
    images = np.asarray(images)
    coordinates = coordinates_2d(coordinates)
    if images.ndim != 3 or coordinates.ndim != 4 or len(images) != len(coordinates):
        raise ValueError("Expected images (F,H,W) and trajectories (F,L,K,2)")
    if not np.isfinite(noise_std_per_channel) or noise_std_per_channel < 0:
        raise ValueError("Noise SD must be finite and nonnegative")
    if noise_std_per_channel > 0 and noise_seed is None:
        raise ValueError("A noise seed is required for reproducible noisy acquisitions")
    result = np.empty(coordinates.shape[:-1], complex)
    for frame, image in enumerate(images):
        operator = NufftOperator(
            image.shape, coordinates[frame].reshape(-1, 2), oversampling, kernel_width
        )
        result[frame] = operator.forward(image).reshape(result.shape[1:])
    if noise_std_per_channel:
        rng = np.random.default_rng(noise_seed)
        result += noise_std_per_channel * (
            rng.normal(size=result.shape) + 1j * rng.normal(size=result.shape)
        )
    return AcquisitionResult(
        result,
        coordinates.copy(),
        images.shape[1:],
        oversampling,
        kernel_width,
        noise_std_per_channel,
        noise_seed,
    )
