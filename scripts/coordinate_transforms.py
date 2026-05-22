"""Coordinate transforms for radar-camera sensor fusion.

This module provides utilities to move points across coordinate frames:
- Radar sensor frame -> ego vehicle frame
- Ego vehicle frame -> camera sensor frame
- Camera 3D frame -> 2D image plane
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from pyquaternion import Quaternion


def quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a nuScenes quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Args:
        quaternion: Quaternion in nuScenes ordering [w, x, y, z].

    Returns:
        3x3 rotation matrix.
    """
    return Quaternion(quaternion).rotation_matrix


def radar_to_ego(points: np.ndarray, calibration: Dict[str, Any]) -> np.ndarray:
    """Transform radar points from sensor frame to ego vehicle frame.

    The calibrated sensor record encodes a rigid transform where:
    p_ego = R * p_sensor + t

    Args:
        points: Nx3 radar points in sensor frame.
        calibration: Calibrated sensor record with rotation and translation.

    Returns:
        Nx3 points in ego vehicle frame.
    """
    rotation = quaternion_to_rotation_matrix(calibration["rotation"])
    translation = np.asarray(calibration["translation"], dtype=float)
    points = np.asarray(points, dtype=float)
    return points @ rotation.T + translation


def ego_to_camera(points: np.ndarray, calibration: Dict[str, Any]) -> np.ndarray:
    """Transform ego vehicle frame points into the camera sensor frame.

    This inverts the calibrated sensor transform:
    p_sensor = R^{-1} * (p_ego - t)

    Args:
        points: Nx3 points in ego vehicle frame.
        calibration: Calibrated sensor record with rotation and translation.

    Returns:
        Nx3 points in camera sensor frame.
    """
    rotation = quaternion_to_rotation_matrix(calibration["rotation"])
    translation = np.asarray(calibration["translation"], dtype=float)
    points = np.asarray(points, dtype=float)
    return (points - translation) @ rotation


def camera_to_image(
    points: np.ndarray, intrinsic: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points onto the 2D image plane.

    Uses a pinhole camera model:
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    Points with Z <= 0 are behind the camera and are marked invalid.

    Args:
        points: Nx3 points in the camera frame.
        intrinsic: 3x3 camera intrinsics matrix.

    Returns:
        Tuple of (Nx2 pixel coordinates, Nx1 boolean mask for valid points).
        Invalid points have NaN pixel coordinates.
    """
    points = np.asarray(points, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)

    z = points[:, 2]
    valid = z > 0

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    safe_z = np.where(valid, z, 1.0)
    u = fx * (points[:, 0] / safe_z) + cx
    v = fy * (points[:, 1] / safe_z) + cy

    pixels = np.column_stack([u, v])
    pixels[~valid] = np.nan
    return pixels, valid
