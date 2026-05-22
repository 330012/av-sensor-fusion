"""Unit tests for radar projection pipeline."""

from __future__ import annotations

import numpy as np

from scripts.radar_projection import project_radar_to_camera


def _identity_calib() -> dict:
    return {
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0],
        "camera_intrinsic": np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "image_width": 10,
        "image_height": 10,
    }


def test_valid_points_within_bounds_pass_through() -> None:
    radar_points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    radar_calib = _identity_calib()
    ego_pose_radar = _identity_calib()
    ego_pose_camera = _identity_calib()
    camera_calib = _identity_calib()

    pixels, distances = project_radar_to_camera(
        radar_points, radar_calib, ego_pose_radar, ego_pose_camera, camera_calib
    )

    assert pixels.shape == (2, 2)
    assert np.allclose(pixels, np.array([[1.0, 1.0], [1.0, 1.0]]), atol=1e-6)
    assert distances.shape == (2,)


def test_points_outside_image_bounds_filtered() -> None:
    radar_points = np.array([[1.0, 1.0, 1.0], [20.0, 1.0, 1.0]])

    radar_calib = _identity_calib()
    ego_pose_radar = _identity_calib()
    ego_pose_camera = _identity_calib()
    camera_calib = _identity_calib()

    pixels, distances = project_radar_to_camera(
        radar_points, radar_calib, ego_pose_radar, ego_pose_camera, camera_calib
    )

    assert pixels.shape == (1, 2)
    assert distances.shape == (1,)


def test_nan_points_removed() -> None:
    radar_points = np.array([[1.0, 1.0, 1.0], [np.nan, 1.0, 1.0]])

    radar_calib = _identity_calib()
    ego_pose_radar = _identity_calib()
    ego_pose_camera = _identity_calib()
    camera_calib = _identity_calib()

    pixels, distances = project_radar_to_camera(
        radar_points, radar_calib, ego_pose_radar, ego_pose_camera, camera_calib
    )

    assert pixels.shape == (1, 2)
    assert distances.shape == (1,)
