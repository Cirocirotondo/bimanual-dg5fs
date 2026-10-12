# Sourced by run.sh / calibrate_camera.sh: writes copies of the teleop pose-estimation
# configs with @TELEOP_DIR@ replaced by the absolute teleoperation directory.
TELEOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ESTIMATOR_DIR="$TELEOP_DIR/pose_estimation/tag-pose-estimation"
RESOLVED_DIR="$(mktemp -d)"
for f in camera.json pose_estimation.json; do
  sed "s|@TELEOP_DIR@|$TELEOP_DIR|g; s|$TELEOP_DIR/config/camera.json|$RESOLVED_DIR/camera.json|g" \
    "$TELEOP_DIR/config/$f" > "$RESOLVED_DIR/$f"
done
SERIAL="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['cameras'][0]['serial_number'])" "$RESOLVED_DIR/camera.json")"
if [ "$SERIAL" = "SET_ME" ]; then
  echo "Set the camera serial_number in $TELEOP_DIR/config/camera.json first" >&2
  echo "(list connected RealSense cameras with: rs-enumerate-devices -s)" >&2
  exit 1
fi
[ -x "$ESTIMATOR_DIR/.venv/bin/python" ] || { echo "Run pose_estimation/setup.sh first" >&2; exit 1; }
