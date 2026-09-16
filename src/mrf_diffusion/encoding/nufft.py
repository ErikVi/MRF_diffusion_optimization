"""Unitary-normalized CPU NUFFT sampling operator; no implicit reconstruction."""

from dataclasses import dataclass
import numpy as np
from ._validation import coordinates_2d, image_shape_2d, sigpy_module
from .trajectory import to_sigpy_coordinates


def _complex_array(values, shape):
    values = np.asarray(values)
    if values.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {values.shape}")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("image/samples must contain finite numeric values")
    # Preserve complex64/128; float64 and integer input use complex128.
    dtype = (
        np.complex64
        if values.dtype in (np.dtype("float32"), np.dtype("complex64"))
        else np.complex128
    )
    return np.array(values, dtype=dtype, copy=True)


@dataclass(frozen=True, eq=False)
class NufftOperator:
    """One non-Cartesian acquisition of a 2-D image.

    image_shape=(H,W); coordinates=(K,2) in cycles/pixel, ordered (ky,kx).
    forward maps (H,W)->(K,); adjoint maps (K,)->(H,W). No broadcasting/batching.
    Ideal A[k,r]=exp(-2*pi*i*k.r)/sqrt(H*W), r=index-floor(shape/2).
    SigPy approximates this operator with Kaiser-Bessel interpolation. Precision
    is complex64 or complex128 matching input precision (integers -> complex128).
    No DCF, gain fitting, magnitude conversion or phase correction is implicit.
    This host NumPy/SigPy API is not a JAX differentiable operation.
    """

    image_shape: tuple
    coordinates: np.ndarray
    oversampling: float = 1.25
    kernel_width: float = 4.0

    def __post_init__(self):
        shape = image_shape_2d(self.image_shape)
        coordinates = coordinates_2d(self.coordinates, single=True).copy()
        if not np.isfinite(self.oversampling) or self.oversampling <= 1:
            raise ValueError("oversampling must be finite and greater than one")
        if not np.isfinite(self.kernel_width) or self.kernel_width < 2:
            raise ValueError("kernel_width must be finite and at least two")
        coordinates.setflags(write=False)
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "coordinates", coordinates)

    def forward(self, image):
        """Encode a complex image into unweighted complex k-space samples."""
        image = _complex_array(image, self.image_shape)
        sp = sigpy_module()
        return sp.nufft(
            image,
            to_sigpy_coordinates(self.coordinates, self.image_shape),
            oversamp=self.oversampling,
            width=self.kernel_width,
        )

    def adjoint(self, samples):
        """Apply A^H to complex samples. An adjoint is not generally an inverse."""
        samples = _complex_array(samples, (len(self.coordinates),))
        sp = sigpy_module()
        return sp.nufft_adjoint(
            samples,
            to_sigpy_coordinates(self.coordinates, self.image_shape),
            oshape=self.image_shape,
            oversamp=self.oversampling,
            width=self.kernel_width,
        )
