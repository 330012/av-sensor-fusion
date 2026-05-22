"""Unit tests for coordinate transform helpers."""

from __future__ import annotations

import numpy as np
from pyquaternion import Quaternion

from scripts.coordinate_transforms import (
    camera_to_image,
    ego_to_camera,
    radar_to_ego,
)


def test_identity_transform_returns_input() -> None:
    calibration = {"rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]}
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]])

    radar_out = radar_to_ego(points, calibration)
    camera_out = ego_to_camera(points, calibration)

    assert np.allclose(radar_out, points)
    assert np.allclose(camera_out, points)


def test_known_radar_point_transform() -> None:
    quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=np.pi / 2)
    calibration = {
        "rotation": [quat.w, quat.x, quat.y, quat.z],
        "translation": [1.0, 2.0, 0.0],
    }
    points = np.array([[10.0, 0.0, 0.0]])

    result = radar_to_ego(points, calibration)
    expected = np.array([[1.0, 12.0, 0.0]])

    assert np.allclose(result, expected)


def test_camera_to_image_filters_points_behind_camera() -> None:
    points = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, -1.0]])
    intrinsic = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    pixels, valid = camera_to_image(points, intrinsic)

    assert valid.tolist() == [True, False]
    assert np.allclose(pixels[0], [1.0, 1.0])
    assert np.isnan(pixels[1]).all()
