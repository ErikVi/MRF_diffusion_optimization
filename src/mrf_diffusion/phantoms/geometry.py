"""Image support and labels, independent of tissue values and simulation."""

import numpy as np


def checkerboard_labels(image_shape, tile_shape):
    """Return binary (H,W) labels; partial tiles at boundaries are allowed."""
    if (
        len(image_shape) != 2
        or len(tile_shape) != 2
        or any(
            isinstance(n, bool) or int(n) != n or n < 1
            for n in (*image_shape, *tile_shape)
        )
    ):
        raise ValueError("image/tile shapes must be positive integer pairs")
    rows, columns = np.indices(tuple(map(int, image_shape)))
    return ((rows // tile_shape[0] + columns // tile_shape[1]) % 2).astype(int)


def circular_support(image_shape, radius_pixels=None):
    """Boolean (H,W) support centered at floor(shape/2); None includes all pixels."""
    checkerboard_labels(image_shape, (1, 1))
    if radius_pixels is None:
        return np.ones(image_shape, bool)
    if not np.isfinite(radius_pixels) or radius_pixels < 0:
        raise ValueError("radius_pixels must be finite and nonnegative")
    row, column = np.indices(image_shape)
    return (row - image_shape[0] // 2) ** 2 + (
        column - image_shape[1] // 2
    ) ** 2 <= radius_pixels**2


def quadratic_object_phase_map(image_shape, edge_phase_rad=0.0):
    """Return (H,W) object phase in radians, NOT a complex phasor or RF train.

    Quadratic radius normalized by the squared distance to the left edge along
    the central row. Odd-size behavior matches the reference field when
    edge_phase_rad=2*pi*order. Even and rectangular sizes retain exact shape.
    """
    checkerboard_labels(image_shape, (1, 1))
    if not np.isfinite(edge_phase_rad):
        raise ValueError("edge_phase_rad must be finite")
    row, column = np.indices(image_shape)
    radius_squared = (row - image_shape[0] // 2) ** 2 + (
        column - image_shape[1] // 2
    ) ** 2
    return edge_phase_rad * radius_squared / max((image_shape[1] // 2) ** 2, 1)
