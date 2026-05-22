"""Dataset loading helpers for nuScenes camera and radar data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

from scripts.config import CAMERA_CHANNEL, DATASET_VERSION, DATA_ROOT, RADAR_CHANNEL


def load_nuscenes() -> NuScenes:
    """Initialize and return a nuScenes dataset instance.

    Returns:
        NuScenes: Initialized dataset instance pointing at DATA_ROOT.
    """
    return NuScenes(version=DATASET_VERSION, dataroot=str(DATA_ROOT), verbose=False)


def get_sample(nusc: NuScenes, index: int) -> Dict[str, Any]:
    """Return a sample dictionary by index.

    Args:
        nusc: Initialized nuScenes dataset.
        index: Zero-based sample index.

    Returns:
        Sample record dictionary.
    """
    return nusc.sample[index]


def get_camera_data(nusc: NuScenes, sample: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    """Return image path and camera calibration for a sample.

    Args:
        nusc: Initialized nuScenes dataset.
        sample: Sample record dictionary.

    Returns:
        Tuple of image path and calibrated sensor record.
    """
    camera_token = sample["data"][CAMERA_CHANNEL]
    camera_sample = nusc.get("sample_data", camera_token)
    calib = nusc.get("calibrated_sensor", camera_sample["calibrated_sensor_token"])
    image_path = Path(nusc.get_sample_data_path(camera_token))
    return image_path, calib


def get_radar_data(
    nusc: NuScenes, sample: Dict[str, Any]
) -> Tuple[RadarPointCloud, Dict[str, Any]]:
    """Return radar point cloud and radar calibration for a sample.

    Args:
        nusc: Initialized nuScenes dataset.
        sample: Sample record dictionary.

    Returns:
        Tuple of radar point cloud and calibrated sensor record.
    """
    radar_token = sample["data"][RADAR_CHANNEL]
    radar_sample = nusc.get("sample_data", radar_token)
    calib = nusc.get("calibrated_sensor", radar_sample["calibrated_sensor_token"])
    radar_path = nusc.get_sample_data_path(radar_token)
    radar_points = RadarPointCloud.from_file(radar_path)
    return radar_points, calib
