"""Visualization utilities for camera-radar fusion outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent_dir(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def plot_radar_overlay(
    image: np.ndarray,
    pixel_coords: np.ndarray,
    distances: np.ndarray,
    output_path: Path,
    title: str = "Camera + Radar Overlay",
) -> None:
    """Plot a camera image with radar points overlaid and colored by distance.

    Args:
        image: HxWxC image array.
        pixel_coords: Nx2 pixel coordinates.
        distances: Nx1 distances from the camera origin (meters).
        output_path: Path to save the figure.
        title: Plot title.
    """
    _ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(image)
    scatter = ax.scatter(
        pixel_coords[:, 0],
        pixel_coords[:, 1],
        c=distances,
        cmap="jet_r",
        s=12,
        alpha=0.85,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Distance (m)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radar_birds_eye(
    radar_points: np.ndarray,
    output_path: Path,
    title: str = "Radar Bird's-Eye View",
) -> None:
    """Plot radar points in a top-down bird's-eye view.

    Args:
        radar_points: Nx3 radar points in the radar sensor frame.
        output_path: Path to save the figure.
        title: Plot title.
    """
    _ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(radar_points[:, 0], radar_points[:, 1], s=6, alpha=0.7)
    ax.scatter([0.0], [0.0], s=80, c="red", marker="x", label="Ego")
    ax.set_title(title)
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.axis("equal")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_range_histogram(distances: np.ndarray, output_path: Path) -> None:
    """Plot a histogram of radar detection ranges.

    Args:
        distances: Nx1 distances from the camera origin (meters).
        output_path: Path to save the figure.
    """
    _ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(distances, bins=20, range=(0, 100), color="#2f4f8f", alpha=0.85)
    ax.set_title("Radar Detection Range Distribution")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
