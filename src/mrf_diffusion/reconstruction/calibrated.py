"""Explicit optional spatial operator gain calibration, independent of fitting."""

import numpy as np
from mrf_diffusion.encoding.series import acquire_image_series
from .series import reconstruct_acquisition_frames


def reconstruct_with_impulse_gain(
    acquisition,
    *,
    density_compensation="none",
    pipe_menon_iterations=30,
    unit_impulse_gain=False,
):
    """Return complex frames and per-frame applied divisor.

    Optional calibration divides A^H W y by the center value of A^H W A delta.
    Uses only the operator, never phantom truth or fitted density. This corrects
    sampling gain, not PSF blurring/aliasing; it does not make the adjoint an inverse.
    """
    kwargs = dict(
        density_compensation=density_compensation,
        pipe_menon_iterations=pipe_menon_iterations,
    )
    frames = reconstruct_acquisition_frames(acquisition, **kwargs)
    gains = np.ones(len(frames))
    if unit_impulse_gain:
        impulse = np.zeros_like(frames)
        center = tuple(size // 2 for size in acquisition.image_shape)
        impulse[(slice(None), *center)] = 1
        encoded = acquire_image_series(
            impulse,
            acquisition.coordinates,
            oversampling=acquisition.oversampling,
            kernel_width=acquisition.kernel_width,
        )
        response = reconstruct_acquisition_frames(encoded, **kwargs)[
            (slice(None), *center)
        ]
        if (
            np.any(response.real <= 0)
            or not np.all(np.isfinite(response))
            or not np.allclose(response.imag, 0, atol=1e-12)
        ):
            raise ValueError("Invalid central impulse gain")
        gains = response.real
        frames = frames / gains[:, None, None]
    return frames, gains
