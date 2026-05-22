AV Sensor Fusion
================

Portfolio project demonstrating camera-radar sensor fusion on the nuScenes
mini dataset. The pipeline loads synchronized camera and radar samples,
projects radar point clouds onto camera images, computes coverage statistics,
and visualizes the results.

Stack
-----
- Python 3.11
- NumPy, SciPy, pandas, matplotlib
- OpenCV, Open3D
- nuScenes-devkit

Project Structure
-----------------
- data/           nuScenes dataset (gitignored)
- notebooks/      analysis notebooks
- outputs/        generated figures and stats (gitignored)
- scripts/        pipeline modules
- tests/          unit tests

Setup
-----
1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:

	pip install -r requirements.txt

3. Install nuScenes devkit without its pinned dependencies:

	pip install --no-deps nuscenes-devkit==1.1.11

Dataset
-------
Download the nuScenes mini dataset and place it under data/nuscenes/.
The path and dataset version are configured in scripts/config.py.

Usage
-----
- Notebooks live under notebooks/.
- The CLI entry point will be scripts/main.py (added in later phases).

License
-------
MIT. See LICENSE.
