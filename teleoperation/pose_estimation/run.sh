#!/usr/bin/env bash
# Runs the Wrist board pose estimator: both Wrist boards (right = board 0,
# left = board 1) in the World board frame, published on tcp 5557. Cameras in
# priority order from config/camera.json (overhead first, the one below as
# fallback). One window per camera; add --headless to run without windows.
set -euo pipefail
unset PYTHONPATH
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -x "$HERE/tag-pose-estimation/.venv/bin/python" ] || { echo "Run pose_estimation/setup.sh first" >&2; exit 1; }
[ -f "$HERE/../calibration/T_world_camera.npy" ] || { echo "Calibrate the camera first: pose_estimation/calibrate_camera.sh" >&2; exit 1; }
exec "$HERE/tag-pose-estimation/.venv/bin/python" "$HERE/run_estimator.py" "$@"
