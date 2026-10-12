"""Extrinsic calibration of the overhead camera against the World board.

Shows the camera image with the *world* axes the teleoperation uses drawn on
the World board. Place the board so that, seen from where you stand to
teleoperate,
    red   (world x) points to your LEFT,
    green (world y) points TOWARDS YOU,
    blue  (world z) points UP,
then press  c  in the window to save calibration/T_world_camera.npy (the pose
is averaged over one second of frames), or  q  to quit without saving.

The world frame is the one of tag-pose-estimation's
camera_extrinsic_calibration_single.py (ChArUco board frame rotated by
R_z(90) R_x(180)). That script uses ArUco functions removed in OpenCV 4.11,
which its own requirements pin, so this replaces it.

Run with the pose estimator's environment (calibrate_camera.sh does this):
    pose_estimation/tag-pose-estimation/.venv/bin/python pose_estimation/calibrate_camera.py
    ... --snapshot out.jpg     # save one annotated frame and exit (no calibration)
    ... --capture              # no window: calibrate as soon as the board is seen
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tag-pose-estimation"))
from tag_pose_estimation.camera_wrappers import RealSenseCamera  # noqa: E402
from tag_pose_estimation.utils import load_charuco_board_from_json  # noqa: E402

TELEOP_DIR = HERE.parent
OUTPUT = TELEOP_DIR / "calibration" / "T_world_camera.npy"
R_X_180 = np.diag([1.0, -1.0, -1.0])
R_Z_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
R_WORLD_FROM_BOARD = R_Z_90 @ R_X_180
AVERAGE_S = 1.0


class BoardTracker:
    def __init__(self, camera):
        self.camera = camera
        self.board, dictionary = load_charuco_board_from_json(
            str(TELEOP_DIR / "config" / "boards" / "world_charuco_5x7_35mm_26mm.json"))
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def camera_from_board(self, frame):
        """(4x4 T_camera_board or None, status text); draws detections on frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
        if ids is None or len(ids) < 6:
            return None, f"World board: {0 if ids is None else len(ids)} ChArUco corners (need 6)"
        object_points, image_points = self.board.matchImagePoints(corners, ids)
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera.camera_matrix,
                                      self.camera.dist_coeffs)
        if not ok:
            return None, "World board pose not found"
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(rvec)[0]
        transform[:3, 3] = tvec.ravel()
        return transform, f"World board: {len(ids)} ChArUco corners"

    def draw_world_axes(self, frame, camera_from_board):
        camera_from_world = camera_from_board[:3, :3] @ R_WORLD_FROM_BOARD.T
        origin = camera_from_board[:3, 3]
        k, d = self.camera.camera_matrix, self.camera.dist_coeffs
        cv2.drawFrameAxes(frame, k, d, cv2.Rodrigues(camera_from_world)[0], origin, 0.12, 4)
        for axis, label in ((0, "x: your LEFT"), (1, "y: TOWARDS YOU"), (2, "z: up")):
            point = camera_from_world[:, axis] * 0.14 + origin
            pixel, _ = cv2.projectPoints(point.reshape(1, 3), np.zeros(3), np.zeros(3), k, d)
            u, v = pixel.ravel().astype(int)
            cv2.putText(frame, label, (u + 5, v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def world_from_camera(camera_from_board_samples):
    """Averages the board poses, then T_world_camera = T_world_board inv(T_camera_board)."""
    rotations = Rotation.from_matrix([t[:3, :3] for t in camera_from_board_samples]).mean()
    camera_from_board = np.eye(4)
    camera_from_board[:3, :3] = rotations.as_matrix()
    camera_from_board[:3, 3] = np.mean([t[:3, 3] for t in camera_from_board_samples], axis=0)
    world_from_board = np.eye(4)
    world_from_board[:3, :3] = R_WORLD_FROM_BOARD
    return world_from_board @ np.linalg.inv(camera_from_board)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", help="save one annotated frame to this path and exit")
    parser.add_argument("--capture", action="store_true",
                        help="no window: start capturing as soon as the board is seen (after 2 s)")
    args = parser.parse_args()

    serial = json.loads((TELEOP_DIR / "config" / "camera.json").read_text())["cameras"][0]["serial_number"]
    camera = RealSenseCamera(serial_number=serial)
    tracker = BoardTracker(camera)
    samples, collecting_until = [], None
    start = time.monotonic()
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            frame = frame.copy()
            pose, status = tracker.camera_from_board(frame)
            if pose is not None:
                tracker.draw_world_axes(frame, pose)
            if collecting_until is not None:
                if pose is not None:
                    samples.append(pose)
                status = f"capturing... {len(samples)} frames"
                if time.monotonic() > collecting_until:
                    break
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            settled = time.monotonic() - start > 2.0   # let auto exposure settle
            if args.snapshot:
                if settled:
                    cv2.imwrite(args.snapshot, frame)
                    print(status)
                    return
                continue
            if args.capture:
                if settled and pose is not None and collecting_until is None:
                    collecting_until = time.monotonic() + AVERAGE_S
                if time.monotonic() - start > 15.0 and collecting_until is None:
                    sys.exit(f"World board not found ({status})")
                continue
            cv2.imshow("World board (c = calibrate, q = quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("quit without saving")
                return
            if key == ord("c") and pose is not None and collecting_until is None:
                collecting_until = time.monotonic() + AVERAGE_S
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(samples) < 10:
        sys.exit(f"only {len(samples)} board poses captured; keep the board fully in view and retry")
    transform = world_from_camera(samples)
    np.save(OUTPUT, transform)
    spread_mm = 1000 * np.std([s[:3, 3] for s in samples], axis=0).max()
    print(f"saved {OUTPUT} from {len(samples)} frames (position spread {spread_mm:.1f} mm)")
    print(f"camera position in the world frame [m]: {transform[:3, 3].round(3).tolist()}")


if __name__ == "__main__":
    main()
