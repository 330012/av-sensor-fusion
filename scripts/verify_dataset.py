"""Quick dataset verification script for nuScenes mini."""

from __future__ import annotations

from pprint import pprint

from scripts.data_loader import get_sample, load_nuscenes


def main() -> None:
    """Load nuScenes and print scene count plus sample 0 metadata."""
    nusc = load_nuscenes()
    print(f"Scenes: {len(nusc.scene)}")
    sample = get_sample(nusc, 0)
    pprint(sample)


if __name__ == "__main__":
    main()
