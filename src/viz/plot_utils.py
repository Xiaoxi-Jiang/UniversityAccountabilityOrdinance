"""Shared plotting helpers for non-interactive figure generation."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import textwrap

cache_dir = Path(tempfile.gettempdir()) / "uoa-matplotlib-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_figure(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")
    return output_path


def wrap_labels(labels: pd.Series, width: int = 20) -> list[str]:
    return [textwrap.fill(str(label), width) for label in labels]


def truncate_labels(values: pd.Series, width: int = 40) -> list[str]:
    return [str(value)[:width] for value in values]


def add_bar_labels(ax: plt.Axes) -> None:
    """Add lightweight value labels to bar charts when readability benefits."""
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=8)
