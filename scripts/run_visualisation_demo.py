"""Run a demo that visualizes camera-radar fusion outputs."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

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
from scripts.visualisation import (
    plot_radar_birds_eye,
    plot_radar_overlay,
    plot_range_histogram,
)


def _get_sample_and_records() -> tuple[object, dict, dict, dict]:
    nusc = load_nuscenes()
    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])

    camera_token = sample["data"][CAMERA_CHANNEL]
    radar_token = sample["data"][RADAR_CHANNEL]

    camera_sample = nusc.get("sample_data", camera_token)
    radar_sample = nusc.get("sample_data", radar_token)

    return nusc, sample, camera_sample, radar_sample


def main() -> None:
    """Load nuScenes mini sample 0 and generate visualizations."""
    nusc, sample, camera_sample, radar_sample = _get_sample_and_records()

    image_path, camera_calib = get_camera_data(nusc, sample)
    radar_cloud, radar_calib = get_radar_data(nusc, sample)

    ego_pose_radar = nusc.get("ego_pose", radar_sample["ego_pose_token"])
    ego_pose_camera = nusc.get("ego_pose", camera_sample["ego_pose_token"])

    camera_calib = {
        **camera_calib,
        "image_width": camera_sample["width"],
        "image_height": camera_sample["height"],
    }

    radar_points = radar_cloud.points[:3, :].T
    pixels, distances = project_radar_to_camera(
        radar_points,
        radar_calib,
        ego_pose_radar,
        ego_pose_camera,
        camera_calib,
    )

    image = plt.imread(str(image_path))

    figures_dir = OUTPUT_DIR / "figures"
    overlay_path = figures_dir / "camera_radar_overlay.png"
    birds_eye_path = figures_dir / "radar_birds_eye.png"
    hist_path = figures_dir / "radar_range_hist.png"

    plot_radar_overlay(image, pixels, distances, overlay_path)
    plot_radar_birds_eye(radar_points, birds_eye_path)
    plot_range_histogram(distances, hist_path)

    print("Saved figures:")
    print(overlay_path)
    print(birds_eye_path)
    print(hist_path)


if __name__ == "__main__":
    main()
