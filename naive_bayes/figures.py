"""The figure the README leads with."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PLAIN_COLOUR = "#cf222e"
SMOOTHED_COLOUR = "#1a7f37"


def smoothing_comparison(metrics_path: Path, output_path: Path) -> Path:
    """Per-category F1 with and without additive smoothing, side by side."""
    metrics = json.loads(Path(metrics_path).read_text())
    categories = metrics["categories"]
    plain = metrics["without_smoothing"]
    smoothed = metrics["additive_smoothing_alpha_1"]

    positions = np.arange(len(categories))
    figure, axis = plt.subplots(figsize=(9, 4.2))
    axis.bar(positions - 0.2, plain["f1"], 0.4, label="no smoothing", color=PLAIN_COLOUR)
    axis.bar(positions + 0.2, smoothed["f1"], 0.4, label="alpha = 1", color=SMOOTHED_COLOUR)

    for x, (before, after) in enumerate(zip(plain["f1"], smoothed["f1"])):
        axis.text(x + 0.2, after + 0.4, f"+{after - before:.1f}", ha="center", fontsize=9,
                  color=SMOOTHED_COLOUR)

    axis.set_xticks(positions)
    axis.set_xticklabels([c.replace(" and ", "\nand ") for c in categories], fontsize=10)
    axis.set_ylabel("F1 (%)")
    axis.set_ylim(85, 100)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="lower right")
    axis.set_title(
        f"Additive smoothing raises accuracy from {plain['accuracy']}% "
        f"to {smoothed['accuracy']}%",
        fontsize=12,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path
