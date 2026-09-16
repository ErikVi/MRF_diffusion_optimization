"""Comparable quantitative triptychs and sampling curves; export PNG and PDF."""

from pathlib import Path
import numpy as np

LABELS = {
    "t1_ms": "T1 [ms]",
    "t2_ms": "T2 [ms]",
    "md_mm2_per_s": "MD [mm²/s]",
    "fa": "FA",
    "proton_density": "Relative proton density",
}


def plot_parameter_triptychs(directory, truth, recovered, support, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, actual in truth.items():
        fitted = recovered[name]
        error = fitted - actual
        limits = np.asarray(actual)[support]
        vmin, vmax = limits.min(), limits.max()
        if vmax == vmin:
            vmin, vmax = vmin - max(abs(vmin) * 0.05, 1e-6), vmax + max(
                abs(vmax) * 0.05, 1e-6
            )
        finite_error = np.abs(error[support & np.isfinite(error)])
        extent = max(float(finite_error.max()) if finite_error.size else 0, 1e-12)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
        for ax, image, caption, cmap, low, high in zip(
            axes,
            (actual, fitted, error),
            ("Ground truth", "Recovered", "Recovered − truth"),
            ("viridis", "viridis", "RdBu_r"),
            (vmin, vmin, -extent),
            (vmax, vmax, extent),
        ):
            artist = ax.imshow(
                np.ma.masked_where(~support | ~np.isfinite(image), image),
                cmap=cmap,
                vmin=low,
                vmax=high,
                interpolation="nearest",
            )
            ax.set_title(caption)
            ax.set_xlabel("Column [pixel]")
            ax.set_ylabel("Row [pixel]" if ax is axes[0] else "")
            fig.colorbar(artist, ax=ax, shrink=0.8)
        fig.suptitle(title + "\n" + LABELS[name])
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_sampling_comparison(directory, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Noise realizations remain independent records; curves show mean RMSE only.
    for parameter, label in LABELS.items():
        noises = sorted({r["noise_std_per_channel"] for r in records})
        fig, axes = plt.subplots(
            1,
            len(noises),
            figsize=(5 * len(noises), 3.7),
            squeeze=False,
            sharey=True,
            layout="constrained",
        )
        for ax, noise in zip(axes[0], noises):
            for sequence in sorted({r["sequence"] for r in records}):
                rows = [
                    r
                    for r in records
                    if r["sequence"] == sequence and r["noise_std_per_channel"] == noise
                ]
                arms = sorted(
                    {r["interleaves"] for r in rows if r["sampling"] == "spiral"}
                )
                values = [
                    np.mean(
                        [
                            r["metrics"]["all"][parameter]["rmse"]
                            for r in rows
                            if r["sampling"] == "spiral" and r["interleaves"] == arm
                        ]
                    )
                    for arm in arms
                ]
                (line,) = ax.plot(arms, values, "o-", label=sequence)
                reference = [
                    r["metrics"]["all"][parameter]["rmse"]
                    for r in rows
                    if r["sampling"] == "cartesian"
                ]
                if reference:
                    ax.axhline(
                        np.mean(reference),
                        color=line.get_color(),
                        linestyle="--",
                        label=sequence + " Cartesian",
                    )
            ax.set(
                xlabel="Spiral interleaves per frame",
                ylabel="RMSE: " + label,
                title=f"K-space SD/channel={noise:g}",
            )
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            ax.set_ylim(bottom=0)
        for extension in ("png", "pdf"):
            fig.savefig(
                Path(directory) / f"comparison_{parameter}.{extension}", dpi=300
            )
        plt.close(fig)
