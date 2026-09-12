"""Legacy tensor phantom; some generated tensors are not positive semidefinite."""

import numpy as np
import jax.numpy as jnp


def make_checkerboard_tensor_phantom(
    base_tensor, n_tiles=11, radius=0.7 / 2 * 121, tile_size=11
):
    height, width = (n_tiles * tile_size, n_tiles * tile_size)
    diffusion_tensor_map = np.zeros((height, width, 3, 3))
    for i in range(n_tiles):
        md_scale = 0.5 + i / (n_tiles - 1)
        for j in range(n_tiles):
            scale_yyzz = 1.0 - j / (n_tiles - 1)
            D = base_tensor.at[1, 1].set(base_tensor[1, 1] * scale_yyzz)
            D = D.at[2, 2].set(base_tensor[2, 2] * scale_yyzz)
            D_scaled = md_scale * D
            y_start, y_end = (i * tile_size, (i + 1) * tile_size)
            x_start, x_end = (j * tile_size, (j + 1) * tile_size)
            diffusion_tensor_map[y_start:y_end, x_start:x_end, :, :] = np.array(
                D_scaled
            )
    y_grid, x_grid = np.ogrid[:height, :width]
    center_y, center_x = (height // 2, width // 2)
    dist_from_center = np.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
    mask = dist_from_center <= radius
    diffusion_tensor_map[~mask] = jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    return diffusion_tensor_map
