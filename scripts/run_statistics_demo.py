"""Run a demo that computes and visualizes radar coverage statistics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate project root with scripts/")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import OUTPUT_DIR
from scripts.data_loader import load_nuscenes
from scripts.statistics import analyse_scene
from scripts.visualisation import plot_range_histogram


def _print_summary(stats: dict) -> None:
    print("Scene statistics")
    print("----------------")
    print(f"Scene: {stats['scene_name']}")
    print(f"Samples processed: {stats['total_samples']}")
    print(f"Average points per frame: {stats['average_points_per_frame']}")
    print(
        f"Coverage: {stats['coverage']['coverage_percentage']}% "
        f"(avg {stats['coverage']['average_coverage_percentage']}%)"
    )
    print("Range stats:")
    for key, value in stats["range_stats"].items():
        print(f"  {key:>6}: {value}")


def _save_json(stats: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)


def _plot_coverage_by_sample(coverage: list[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(coverage)), coverage, marker="o", linewidth=1.8)
    ax.set_title("Camera FOV Coverage by Sample")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_density_by_range(density: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(density.keys())
    counts = [density[label] for label in labels]
    x_positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_positions, counts, color="#3b6fb6")
    ax.set_title("Radar Point Density by Range")
    ax.set_xlabel("Range bin")
    ax.set_ylabel("Point count")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the scene-level statistics demo for scene 0."""
    nusc = load_nuscenes()
    stats = analyse_scene(nusc, scene_index=0)

    _print_summary(stats)

    stats_path = OUTPUT_DIR / "stats" / "scene_0_stats.json"
    _save_json(stats, stats_path)

    figures_dir = OUTPUT_DIR / "figures"
    range_plot_path = figures_dir / "range_distribution_scene.png"
    coverage_plot_path = figures_dir / "coverage_by_sample.png"
    density_plot_path = figures_dir / "density_by_range.png"

    distances = np.asarray(stats["distances"], dtype=float)
    plot_range_histogram(distances, range_plot_path)
    _plot_coverage_by_sample(stats["coverage_by_sample"], coverage_plot_path)
    _plot_density_by_range(stats["density_by_range"], density_plot_path)

    print("\nSaved outputs:")
    print(stats_path)
    print(range_plot_path)
    print(coverage_plot_path)
    print(density_plot_path)


if __name__ == "__main__":
    main()
