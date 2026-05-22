"""Radar point projection pipeline for camera fusion.

This module stitches together coordinate transforms to move radar points from
sensor frame to ego vehicle frame, to camera frame, and finally onto the image
plane. It also provides helpers to load radar point clouds and filter points
that fall outside image bounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud

from scripts.coordinate_transforms import (
    camera_to_image,
    ego_to_camera,
    quaternion_to_rotation_matrix,
    radar_to_ego,
)


def load_radar_pointcloud(file_path: Path) -> np.ndarray:
    """Load a nuScenes radar point cloud file and return Nx3 points.

    Args:
        file_path: Path to the radar point cloud file.

    Returns:
        Nx3 array of radar points in the radar sensor frame.
    """
    radar_points = RadarPointCloud.from_file(str(file_path))
    return radar_points.points[:3, :].T


def filter_valid_points(pixels: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Return a mask of points that fall within image bounds.

    Args:
        pixels: Nx2 array of pixel coordinates.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Boolean mask where True indicates points within image bounds.
    """
    pixels = np.asarray(pixels, dtype=float)
    finite = np.isfinite(pixels).all(axis=1)
    in_x = (pixels[:, 0] >= 0) & (pixels[:, 0] < image_width)
    in_y = (pixels[:, 1] >= 0) & (pixels[:, 1] < image_height)
    return finite & in_x & in_y


def _apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to points: p_out = R * p_in + t."""
    return points @ rotation.T + translation


def _apply_inverse_transform(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply the inverse transform: p_out = R^{-1} * (p_in - t)."""
    return (points - translation) @ rotation


def project_radar_to_camera(
    radar_points: np.ndarray,
    radar_calib: Dict[str, Any],
    ego_pose_radar: Dict[str, Any],
    ego_pose_camera: Dict[str, Any],
    camera_calib: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project radar points onto the camera image plane.

    Pipeline:
        radar frame -> ego frame -> world frame -> ego frame (camera time)
        -> camera frame -> image plane

    Args:
        radar_points: Nx3 radar points in radar sensor frame.
        radar_calib: Calibrated sensor record for radar.
        ego_pose_radar: Ego pose record at radar timestamp.
        ego_pose_camera: Ego pose record at camera timestamp.
        camera_calib: Calibrated sensor record for camera. Must include
            "camera_intrinsic", "image_width", and "image_height".

    Returns:
        Tuple of (Nx2 pixel coordinates, Nx1 distances from camera origin).
        Only valid points are returned.
    """
    radar_points = np.asarray(radar_points, dtype=float)

    ego_points = radar_to_ego(radar_points, radar_calib)

    radar_rot = quaternion_to_rotation_matrix(ego_pose_radar["rotation"])
    radar_trans = np.asarray(ego_pose_radar["translation"], dtype=float)
    world_points = _apply_transform(ego_points, radar_rot, radar_trans)

    cam_rot = quaternion_to_rotation_matrix(ego_pose_camera["rotation"])
    cam_trans = np.asarray(ego_pose_camera["translation"], dtype=float)
    ego_cam_points = _apply_inverse_transform(world_points, cam_rot, cam_trans)

    camera_points = ego_to_camera(ego_cam_points, camera_calib)

    intrinsics = np.asarray(camera_calib["camera_intrinsic"], dtype=float)
    pixels, valid = camera_to_image(camera_points, intrinsics)

    image_width = int(camera_calib["image_width"])
    image_height = int(camera_calib["image_height"])
    in_bounds = filter_valid_points(pixels, image_width, image_height)

    valid_mask = valid & in_bounds
    distances = np.linalg.norm(camera_points, axis=1)

    return pixels[valid_mask], distances[valid_mask]
