"""CLI entry point for running the camera-radar fusion pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

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

from scripts.config import CAMERA_CHANNEL, OUTPUT_DIR, RADAR_CHANNEL
from scripts.data_loader import get_camera_data, get_radar_data, load_nuscenes
from scripts.radar_projection import project_radar_to_camera
from scripts.statistics import (
    compute_fov_overlap,
    compute_point_density_by_range,
    compute_range_stats,
)
from scripts.visualisation import plot_radar_birds_eye, plot_radar_overlay, plot_range_histogram


def _iter_samples(nusc: Any, scene_index: int, num_frames: int) -> List[Dict[str, Any]]:
    scene = nusc.scene[scene_index]
    sample_token = scene["first_sample_token"]
    samples: List[Dict[str, Any]] = []

    while sample_token and len(samples) < num_frames:
        sample = nusc.get("sample", sample_token)
        samples.append(sample)
        sample_token = sample["next"]

    return samples


def _save_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_pipeline(scene_index: int, num_frames: int, output_dir: Path) -> Dict[str, Any]:
    """Run the fusion pipeline for a scene and return summary stats.

    Args:
        scene_index: Index of the scene to process.
        num_frames: Number of samples to process.
        output_dir: Output directory for figures and stats.

    Returns:
        Summary statistics for the processed samples.
    """
    nusc = load_nuscenes()
    scene = nusc.scene[scene_index]
    samples = _iter_samples(nusc, scene_index, num_frames)

    figures_dir = output_dir / "figures"
    distances_all: List[float] = []
    coverage_by_sample: List[float] = []
    total_points = 0
    total_in_fov = 0

    for idx, sample in enumerate(samples):
        camera_token = sample["data"][CAMERA_CHANNEL]
        radar_token = sample["data"][RADAR_CHANNEL]
        camera_sample = nusc.get("sample_data", camera_token)
        radar_sample = nusc.get("sample_data", radar_token)

        image_path, camera_calib = get_camera_data(nusc, sample)
        radar_cloud, radar_calib = get_radar_data(nusc, sample)

        camera_calib = {
            **camera_calib,
            "image_width": camera_sample["width"],
            "image_height": camera_sample["height"],
        }

        ego_pose_radar = nusc.get("ego_pose", radar_sample["ego_pose_token"])
        ego_pose_camera = nusc.get("ego_pose", camera_sample["ego_pose_token"])

        radar_points = radar_cloud.points[:3, :].T
        pixels, distances = project_radar_to_camera(
            radar_points,
            radar_calib,
            ego_pose_radar,
            ego_pose_camera,
            camera_calib,
        )

        image = plt.imread(str(image_path))
        overlay_path = figures_dir / f"scene_{scene_index}_sample_{idx}_overlay.png"
        plot_radar_overlay(
            image,
            pixels,
            distances,
            overlay_path,
            title=f"Scene {scene['name']} Sample {idx} Overlay",
        )

        if idx == 0:
            birds_eye_path = figures_dir / f"scene_{scene_index}_sample_{idx}_birds_eye.png"
            plot_radar_birds_eye(radar_points, birds_eye_path)

        fov_stats = compute_fov_overlap(
            pixels,
            int(camera_calib["image_width"]),
            int(camera_calib["image_height"]),
            radar_points.shape[0],
        )

        total_points += radar_points.shape[0]
        total_in_fov += fov_stats["points_in_camera_fov"]
        coverage_by_sample.append(fov_stats["coverage_percentage"])
        distances_all.extend(distances.tolist())

    distances_array = np.asarray(distances_all, dtype=float)
    average_points = 0.0 if not samples else total_points / len(samples)
    overall_coverage = 0.0 if total_points == 0 else round(100.0 * total_in_fov / total_points, 2)

    stats = {
        "scene_index": scene_index,
        "scene_name": scene["name"],
        "total_samples": len(samples),
        "average_points_per_frame": round(average_points, 2),
        "coverage": {
            "points_in_camera_fov": total_in_fov,
            "points_outside_fov": max(total_points - total_in_fov, 0),
            "coverage_percentage": overall_coverage,
        },
        "range_stats": compute_range_stats(distances_array),
        "density_by_range": compute_point_density_by_range(distances_array),
        "coverage_by_sample": coverage_by_sample,
    }

    stats_path = output_dir / "stats" / f"scene_{scene_index}_summary.json"
    _save_json(stats, stats_path)

    range_plot_path = figures_dir / f"scene_{scene_index}_range_distribution.png"
    plot_range_histogram(distances_array, range_plot_path)

    print("Saved outputs:")
    print(stats_path)
    print(range_plot_path)
    print(figures_dir)

    return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the AV sensor fusion pipeline.")
    parser.add_argument("--scene-index", type=int, default=0, help="Scene index to process.")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to process.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for figures and stats.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_pipeline(args.scene_index, args.num_frames, args.output_dir)


if __name__ == "__main__":
    main()
