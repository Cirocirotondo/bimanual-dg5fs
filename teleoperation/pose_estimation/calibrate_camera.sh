#!/usr/bin/env bash
# Extrinsic calibration of the overhead camera against the World board.
# A window shows the world axes; press c to save calibration/T_world_camera.npy.
set -euo pipefail
unset PYTHONPATH
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -x "$HERE/tag-pose-estimation/.venv/bin/python" ] || { echo "Run pose_estimation/setup.sh first" >&2; exit 1; }
exec "$HERE/tag-pose-estimation/.venv/bin/python" "$HERE/calibrate_camera.py" "$@"
