#!/usr/bin/env bash
# One-shot setup of the training stack (re-runnable): Isaac Lab pinned to the same
# commit as simtoolreal_newton (Isaac Lab 3.0, Newton backend, kit-less) in
# bimanual-dg5fs/deps/IsaacLab, this repo's packages, and the Limb URDF -> USD conversion.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_BRANCH="develop"
ISAACLAB_COMMIT="a8b4da3c29ae528b39d4b3c9444d782ce58d886d"   # develop, 2026-09-12 (Isaac Lab 3.0 / Newton 1.6.0rc1)
UV="${UV:-uv}"
unset PYTHONPATH
mkdir -p "$ROOT/deps"
if [ ! -d "$ROOT/deps/IsaacLab/.git" ]; then
    git clone --filter=blob:none -b "$ISAACLAB_BRANCH" "$ISAACLAB_REPO" "$ROOT/deps/IsaacLab"
fi
git -C "$ROOT/deps/IsaacLab" fetch --depth 1 origin "$ISAACLAB_COMMIT" 2>/dev/null || true
git -C "$ROOT/deps/IsaacLab" checkout -q "$ISAACLAB_COMMIT"
cd "$ROOT/deps/IsaacLab"
# --inexact keeps the packages installed below on a re-run; viser for the evaluator.
"$UV" sync --inexact --extra importers --extra rsl-rl --extra viser ${ISAACLAB_EXTRAS:-}
PY="$ROOT/deps/IsaacLab/.venv/bin/python"
"$UV" pip install --python "$PY" -e "$ROOT" --no-deps          # bimanual_dg5fs (Scene constants)
"$UV" pip install --python "$PY" -e "$HERE" --no-deps          # bimanual_training
"$UV" pip install --python "$PY" "pytorch_kinematics>=0.7" "mujoco==3.12.0" "yourdfpy==0.0.60" "imageio-ffmpeg==0.6.0"
"$PY" "$HERE/scripts/convert_urdf.py"
echo "Done. Python interpreter: $PY"
