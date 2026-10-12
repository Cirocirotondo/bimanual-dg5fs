#!/usr/bin/env bash
# One-time setup of the tag pose estimator in its own Python 3.10 environment
# (it needs numpy<2, gtsam and a compiled apriltag, which the Twin's Python 3.12
# environment cannot host). Code copied from the UR5 PC: ~/git/robohand/src/tag-pose-estimation.
set -euo pipefail
cd "$(dirname "$0")/tag-pose-estimation"
unset PYTHONPATH
[ -x .venv/bin/python ] || uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
# Like install_apriltag.sh, but built in a path without spaces (apriltag's
# Python-docstring step breaks on "Nando's setup") and for this environment's Python.
if [ ! -d local_apriltag ]; then
  BUILD="$(mktemp -d /tmp/apriltag-build-XXXXXX)"
  git clone --depth 1 https://github.com/AprilRobotics/apriltag "$BUILD/apriltag"
  cmake -S "$BUILD/apriltag" -B "$BUILD/apriltag/build" -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python" -DBUILD_EXAMPLES=OFF -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
  cmake --build "$BUILD/apriltag/build" -j"$(nproc)"
  mv "$BUILD/apriltag" local_apriltag
  rm -rf "$BUILD"
fi
.venv/bin/python -c "from local_apriltag.build.apriltag import apriltag; print('apriltag import OK')"
echo "pose estimator ready: $(pwd)/.venv"
