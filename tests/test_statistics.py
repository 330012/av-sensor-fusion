"""Unit tests for statistics helpers."""

from __future__ import annotations

import numpy as np

from scripts.statistics import (
    compute_fov_overlap,
    compute_point_density_by_range,
    compute_range_stats,
)


def test_compute_range_stats_known_values() -> None:
    distances = np.array([1.0, 2.0, 3.0, 4.0])
    stats = compute_range_stats(distances)

    assert stats["min"] == 1.0, "Min should be 1.0"
    assert stats["max"] == 4.0, "Max should be 4.0"
    assert stats["mean"] == 2.5, "Mean should be 2.5"
    assert stats["median"] == 2.5, "Median should be 2.5"
    assert stats["std"] == 1.12, "Std should round to 1.12"
    assert stats["count"] == 4, "Count should be 4"


def test_compute_fov_overlap_known_values() -> None:
    pixels = np.array([[1.0, 1.0], [5.0, 5.0], [-1.0, 0.0], [np.nan, 1.0]])
    stats = compute_fov_overlap(pixels, image_width=6, image_height=6, total_radar_points=4)

    assert stats["points_in_camera_fov"] == 2, "Two points should be in FOV"
    assert stats["points_outside_fov"] == 2, "Two points should be outside FOV"
    assert stats["coverage_percentage"] == 50.0, "Coverage should be 50%"


def test_compute_point_density_by_range_bins() -> None:
    distances = np.array([5.0, 15.0, 15.0, 25.0, 105.0])
    density = compute_point_density_by_range(distances, bin_size=10.0)

    assert density["0-10m"] == 1, "0-10m bin should have 1 point"
    assert density["10-20m"] == 2, "10-20m bin should have 2 points"
    assert density["20-30m"] == 1, "20-30m bin should have 1 point"
