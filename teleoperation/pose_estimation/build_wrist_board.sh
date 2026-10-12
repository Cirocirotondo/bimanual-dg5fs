#!/usr/bin/env bash
# Measures the tag layout of a Wrist board (the AprilTag cube on a Glove) with
# the overhead camera and writes config/boards/<side>_wrist_board.json.
#
#   pose_estimation/build_wrist_board.sh left      # tags 1, 3, 5
#   pose_estimation/build_wrist_board.sh right     # tags 0, 2, 4
#
# In the window: hold the cube ~40-60 cm under the camera so that TWO faces are
# clearly visible, press SPACE, then y to keep the capture (n to drop it).
# Capture every pair of faces (e.g. 1+3, 3+5, 5+1) from a few angles, ~10
# captures in total, then press ESC to build and save the board.
set -euo pipefail
unset PYTHONPATH
SIDE="${1:?usage: build_wrist_board.sh left|right}"
case "$SIDE" in
  left)  IDS="1,3,5" ;;
  right) IDS="0,2,4" ;;
  *) echo "side must be left or right" >&2; exit 1 ;;
esac
source "$(dirname "$0")/_resolve_configs.sh"
cd "$ESTIMATOR_DIR"
NAME="${SIDE}_wrist_board_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python scripts/build_object_board.py \
  --marker_size 0.0418 --tag_type apriltag --tag_family tag16h5 \
  --allowed_ids "$IDS" --name "$NAME" \
  --camera_config_path "$RESOLVED_DIR/camera.json"
TARGET="$TELEOP_DIR/config/boards/${SIDE}_wrist_board.json"
cp "$TARGET" "$TARGET.$(date +%Y%m%d_%H%M%S).bak"
cp "config/object_boards/$NAME.json" "$TARGET"
echo "saved $TARGET (previous definition kept as .bak)"
