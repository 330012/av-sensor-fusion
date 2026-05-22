"""Project configuration for dataset paths and sensor channels."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "nuscenes"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DATASET_VERSION = "v1.0-mini"
CAMERA_CHANNEL = "CAM_FRONT"
RADAR_CHANNEL = "RADAR_FRONT"
