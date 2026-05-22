"""Unit tests for nuScenes data loader helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.config import DATASET_VERSION, DATA_ROOT
from scripts.data_loader import get_camera_data, get_radar_data, get_sample, load_nuscenes


def _dataset_available() -> bool:
    version_path = DATA_ROOT / DATASET_VERSION
    return DATA_ROOT.exists() and version_path.exists()


@pytest.mark.skipif(not _dataset_available(), reason="nuScenes dataset not available")
def test_load_nuscenes_returns_instance() -> None:
    nusc = load_nuscenes()
    assert hasattr(nusc, "scene"), "NuScenes instance should expose scene records."


@pytest.mark.skipif(not _dataset_available(), reason="nuScenes dataset not available")
def test_get_sample_returns_expected_token() -> None:
    nusc = load_nuscenes()
    sample = get_sample(nusc, 0)
    assert sample["token"] == nusc.sample[0]["token"], "Sample token mismatch."


@pytest.mark.skipif(not _dataset_available(), reason="nuScenes dataset not available")
def test_get_camera_data_returns_path_and_intrinsics() -> None:
    nusc = load_nuscenes()
    sample = get_sample(nusc, 0)
    image_path, calib = get_camera_data(nusc, sample)

    assert Path(image_path).exists(), "Camera image path should exist."
    intrinsics = np.asarray(calib["camera_intrinsic"], dtype=float)
    assert intrinsics.shape == (3, 3), "Camera intrinsics must be 3x3."


@pytest.mark.skipif(not _dataset_available(), reason="nuScenes dataset not available")
def test_get_radar_data_returns_points_and_calibration() -> None:
    nusc = load_nuscenes()
    sample = get_sample(nusc, 0)
    radar_cloud, calib = get_radar_data(nusc, sample)

    assert radar_cloud.points.shape[1] > 0, "Radar point cloud should be non-empty."
    assert "rotation" in calib and "translation" in calib, "Radar calibration missing fields."
