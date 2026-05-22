"""Statistical analysis utilities for radar coverage and range behavior."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from scripts.config import CAMERA_CHANNEL, RADAR_CHANNEL
from scripts.data_loader import get_camera_data, get_radar_data
from scripts.radar_projection import project_radar_to_camera


def _round(value: float) -> float:
    return float(round(value, 2))


def compute_range_stats(distances: np.ndarray) -> Dict[str, float | int]:
    """Compute summary statistics for radar detection distances.

    Args:
        distances: Nx1 distances from the camera origin.

    Returns:
        Dictionary with min, max, mean, median, std, and count.
    """
    distances = np.asarray(distances, dtype=float)
    distances = distances[np.isfinite(distances)]

    if distances.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "count": 0,
        }

    return {
        "min": _round(float(np.min(distances))),
        "max": _round(float(np.max(distances))),
        "mean": _round(float(np.mean(distances))),
        "median": _round(float(np.median(distances))),
        "std": _round(float(np.std(distances))),
        "count": int(distances.size),
    }


def compute_fov_overlap(
    pixel_coords: np.ndarray,
    image_width: int,
    image_height: int,
    total_radar_points: int,
) -> Dict[str, float | int]:
    """Compute how many radar points fall inside the camera field of view.

    Args:
        pixel_coords: Nx2 pixel coordinates in the image plane.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        total_radar_points: Total radar points before filtering.

    Returns:
        Dictionary with in-FOV, out-of-FOV counts and coverage percentage.
    """
    pixel_coords = np.asarray(pixel_coords, dtype=float)
    finite = np.isfinite(pixel_coords).all(axis=1)
    in_x = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_width)
    in_y = (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_height)
    in_fov = int((finite & in_x & in_y).sum())
    total = int(total_radar_points)
    outside = max(total - in_fov, 0)
    coverage = 0.0 if total == 0 else _round(100.0 * in_fov / total)

    return {
        "points_in_camera_fov": in_fov,
        "points_outside_fov": outside,
        "coverage_percentage": coverage,
    }


def compute_point_density_by_range(
    distances: np.ndarray, bin_size: float = 10.0
) -> Dict[str, int]:
    """Bucket radar points into distance bins and count occurrences.

    Args:
        distances: Nx1 distances from the camera origin.
        bin_size: Bin width in meters.

    Returns:
        Dictionary mapping range bins to point counts.
    """
    distances = np.asarray(distances, dtype=float)
    distances = distances[np.isfinite(distances)]

    edges = np.arange(0.0, 100.0 + bin_size, bin_size)
    counts, _ = np.histogram(distances, bins=edges, range=(0.0, 100.0))

    bins: Dict[str, int] = {}
    for start, end, count in zip(edges[:-1], edges[1:], counts):
        label = f"{int(start)}-{int(end)}m"
        bins[label] = int(count)

    return bins


def analyse_scene(nusc: Any, scene_index: int) -> Dict[str, Any]:
    """Aggregate radar coverage statistics across a scene.

    Args:
        nusc: Initialized nuScenes dataset instance.
        scene_index: Scene index to analyze.

    Returns:
        Aggregated statistics across the scene.
    """
    scene = nusc.scene[scene_index]
    sample_token = scene["first_sample_token"]

    distances_all: List[float] = []
    coverage_by_sample: List[float] = []
    total_points = 0
    total_in_fov = 0
    samples_processed = 0

    while sample_token:
        sample = nusc.get("sample", sample_token)
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

        total_radar_points = radar_points.shape[0]
        fov_stats = compute_fov_overlap(
            pixels,
            int(camera_calib["image_width"]),
            int(camera_calib["image_height"]),
            total_radar_points,
        )

        total_points += total_radar_points
        total_in_fov += fov_stats["points_in_camera_fov"]
        coverage_by_sample.append(fov_stats["coverage_percentage"])
        distances_all.extend(distances.tolist())
        samples_processed += 1

        sample_token = sample["next"]

    average_points_per_frame = 0.0 if samples_processed == 0 else total_points / samples_processed
    overall_coverage = 0.0 if total_points == 0 else _round(100.0 * total_in_fov / total_points)

    distances_array = np.asarray(distances_all, dtype=float)

    return {
        "scene_index": scene_index,
        "scene_name": scene["name"],
        "scene_description": scene["description"],
        "total_samples": samples_processed,
        "average_points_per_frame": _round(average_points_per_frame),
        "coverage": {
            "points_in_camera_fov": total_in_fov,
            "points_outside_fov": max(total_points - total_in_fov, 0),
            "coverage_percentage": overall_coverage,
            "average_coverage_percentage": _round(
                float(np.mean(coverage_by_sample)) if coverage_by_sample else 0.0
            ),
        },
        "range_stats": compute_range_stats(distances_array),
        "density_by_range": compute_point_density_by_range(distances_array),
        "coverage_by_sample": coverage_by_sample,
        "distances": distances_all,
        "image_path_example": str(image_path),
    }
