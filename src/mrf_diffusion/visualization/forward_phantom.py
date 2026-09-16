"""Diagnostic truth maps and complex frames, with no parameter-estimation claims."""

from pathlib import Path
import numpy as np


def plot_forward_experiment(
    directory, phantom, series, acquisition, reconstructed, frame_indices
):
    """Save truth maps, magnitude/phase frames, trajectories and k-space magnitude."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def map_plot(values, title, name, cmap="viridis"):
        fig, ax = plt.subplots(figsize=(5, 4))
        artist = ax.imshow(values, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(artist, ax=ax)
        fig.tight_layout()
        fig.savefig(directory / name, dpi=130)
        plt.close(fig)

    maps = [
        (phantom.t1_ms, "Ground truth T1 [ms]", "truth_t1.png"),
        (phantom.t2_ms, "Ground truth T2 [ms]", "truth_t2.png"),
        (phantom.mean_diffusivity_map, "Ground truth MD [mm²/s]", "truth_md.png"),
        (phantom.fractional_anisotropy_map, "Ground truth FA", "truth_fa.png"),
        (
            phantom.proton_density,
            "Ground truth proton density multiplier",
            "truth_density.png",
        ),
        (phantom.object_phase_map, "Object phase [rad]", "object_phase.png"),
    ]
    for values, title, name in maps:
        map_plot(values, title, name)
    for frame in frame_indices:
        if not isinstance(frame, (int, np.integer)) or not 0 <= frame < len(
            series.images
        ):
            raise ValueError("Representative frame index outside image series")
        map_plot(
            np.abs(series.images[frame]),
            f"Signal magnitude, frame {frame}",
            f"signal_magnitude_{frame}.png",
        )
        map_plot(
            np.angle(series.images[frame]),
            f"Signal phase [rad], frame {frame}",
            f"signal_phase_{frame}.png",
            "twilight",
        )
        coords = acquisition.coordinates[frame]
        fig, ax = plt.subplots(figsize=(5, 4))
        artist = ax.scatter(
            coords[..., 1].ravel(),
            coords[..., 0].ravel(),
            c=np.abs(acquisition.kspace[frame]).ravel(),
            s=6,
        )
        ax.set(
            xlabel="kx [cycles/pixel]",
            ylabel="ky [cycles/pixel]",
            title=f"Sampling / k-space magnitude, frame {frame}",
            aspect="equal",
        )
        fig.colorbar(artist, ax=ax, label="|k-space|")
        fig.tight_layout()
        fig.savefig(directory / f"kspace_{frame}.png", dpi=130)
        plt.close(fig)
        if reconstructed is not None:
            map_plot(
                np.abs(reconstructed[frame]),
                f"Adjoint magnitude, frame {frame}",
                f"adjoint_magnitude_{frame}.png",
            )
            map_plot(
                np.angle(reconstructed[frame]),
                f"Adjoint phase [rad], frame {frame}",
                f"adjoint_phase_{frame}.png",
                "twilight",
            )
