"""Report actual calculated quantities without assigning missing FA/MD columns."""

from pathlib import Path
import numpy as np


def plot_train(values, directory, filename, ylabel):
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(np.asarray(values))
    axis.set(xlabel="TR index", ylabel=ylabel)
    axis.grid(True)
    fig.tight_layout()
    fig.savefig(Path(directory) / filename, dpi=300)
    plt.close(fig)


def plot_precision(initial, optimized, directory):
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    labels = ["T1", "T2", "equilibrium M"]  # current objective actually returns 3
    indices = np.arange(len(initial))
    axis.plot(indices, initial, "o-", label="Initial")
    axis.plot(indices, optimized, "o-", label="Optimized")
    axis.set_xticks(indices, labels)
    axis.set_ylabel("Relative standard-deviation bound")
    axis.legend()
    fig.tight_layout()
    fig.savefig(Path(directory) / "precision.png", dpi=300)
    plt.close(fig)


def plot_signal_family(signals, scales, directory):
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(12, 5))
    for signal, color in zip(signals, plt.cm.rainbow(np.linspace(0, 1, len(scales)))):
        axis.plot(np.asarray(signal), color=color)
    axis.set(
        xlabel="Sample index",
        ylabel="Signal magnitude",
        title="Diffusion template scale sweep [mm²/s]",
    )
    fig.tight_layout()
    fig.savefig(Path(directory) / "signals.png", dpi=300)
    plt.close(fig)


def plot_phase_comparison(fractions, results, baseline, directory):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, label in enumerate(["T1", "T2", "equilibrium M", "Sum"]):
        ax = axes.flat[i]
        for method, values in results.items():
            values = np.asarray(values)
            ax.plot(
                fractions, values[:, i] if i < 3 else values.sum(axis=1), label=method
            )
        ax.axhline(
            baseline[i] if i < 3 else np.sum(baseline),
            linestyle="--",
            label="Legacy baseline",
        )
        ax.set(
            title=label,
            xlabel="Phase fraction",
            ylabel="Relative standard-deviation bound",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(directory) / "phase_comparison.png", dpi=300)
    plt.close(fig)
