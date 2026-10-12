"""Wrist board pose estimator with a main camera and fallback cameras.

Cameras are listed in priority order in config/camera.json (the overhead D435i
first, then the D455 looking up from below). Every frame of the main camera is
searched for both Wrist boards; only the boards it does not see are looked for
in the next camera's latest frame, and so on. One message per main-camera
frame is published on the pose estimator port, in the World board frame:

    {"timestamp", "poses": {board index: {"position", "rotation_matrix",
     "confidence", "tags", "camera", "alternatives": [{"position", "rotation_matrix"}]}}}

Board poses use tag_pose_estimation/robust_board_pose.py (IPPE per tag; the two
poses of an ambiguous single tag are both published, the Glove IMU picks one).

Run with the pose estimator's environment (run.sh does this):
    pose_estimation/tag-pose-estimation/.venv/bin/python pose_estimation/run_estimator.py [--headless]

A window per camera shows its image with the axes of the boards taken from it
(press q in a window to quit); --headless runs without windows.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

HERE = Path(__file__).resolve().parent
TELEOP_DIR = HERE.parent
sys.path.insert(0, str(HERE / "tag-pose-estimation"))
from tag_pose_estimation.camera_wrappers import RealSenseCamera  # noqa: E402
from tag_pose_estimation.detector_wrappers import AprilTagDetectorWrapper  # noqa: E402
from tag_pose_estimation.robust_board_pose import estimate_board_pose  # noqa: E402

STALE_SECONDARY_S = 0.1     # a fallback frame older than this is not used


def resolve(path):
    return Path(str(path).replace("@TELEOP_DIR@", str(TELEOP_DIR)))


class Camera:
    def __init__(self, config, main):
        self.name = config.get("name", config["serial_number"])
        self.device = RealSenseCamera(extrinsic_calibration_path=str(resolve(config["extrinsic_calibration_file"])),
                                      serial_number=config["serial_number"])
        self.world_from_camera = self.device.homogeneous_transform
        self.frame, self.stamp = None, None
        self.previous = {}           # board index -> (rvec, tvec) in this camera's frame
        # Fallback cameras solve each tag in its own frame (robust_board_pose.py:
        # OpenCV's IPPE fails on tags lying on z = const planes of the board).
        self.local_tag_frame = not main
        # Optional local contrast enhancement before tag detection ("clahe": true in
        # config/camera.json). For the camera below, which sees the cubes against
        # bright ceiling lights, it raised the frames with a cube found from ~60 %
        # to ~90 % at the same 3 ms exposure.
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if config.get("clahe") else None

    def gray(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray) if self.clahe is not None else gray

    def poll(self):
        frame = self.device.get_frame()
        if frame is not None:
            self.frame, self.stamp = frame, time.monotonic()
        return frame is not None

    def release(self):
        self.device.release()


def to_world(camera, rvec, tvec):
    camera_from_board = np.eye(4)
    camera_from_board[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_from_board[:3, 3] = np.ravel(tvec)
    return camera.world_from_camera @ camera_from_board


def board_poses(camera, gray, detector, boards, wanted):
    """{board index: published pose} for the wanted boards seen in this frame."""
    corners, ids = detector.detect(gray)
    if ids is None:
        return {}
    image = {int(i): c[0] for c, i in zip(corners, ids.ravel())}
    found = {}
    for index in wanted:
        tags = {t: obj for t, obj in boards[index].items() if t in image}
        if not tags:
            continue
        previous = camera.previous.get(index, (None, None))
        result = estimate_board_pose(tags, {t: image[t] for t in tags}, camera.device.camera_matrix,
                                     camera.device.dist_coeffs, previous[0], previous[1],
                                     local_tag_frame=camera.local_tag_frame)
        if result is None:
            continue
        rvec, tvec, used, error, alternatives = result
        camera.previous[index] = (rvec, tvec)
        world = to_world(camera, rvec, tvec)
        found[index] = {
            "position": world[:3, 3].tolist(),
            "rotation_matrix": world[:3, :3].tolist(),
            "confidence": max(0.0, 1.0 - error / 5.0),
            "tags": [int(t) for t in used],
            "camera": camera.name,
            "alternatives": [{"position": w[:3, 3].tolist(), "rotation_matrix": w[:3, :3].tolist()}
                             for w in (to_world(camera, r, t) for r, t in alternatives)],
            "_draw": (rvec, tvec),
        }
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="no camera windows")
    args = parser.parse_args()

    config = json.loads((TELEOP_DIR / "config" / "pose_estimation.json").read_text())
    camera_configs = json.loads((TELEOP_DIR / "config" / "camera.json").read_text())["cameras"]
    boards = {}
    for index, path in enumerate(config["object_board_definitions"]):
        definition = json.loads(resolve(path).read_text())
        boards[str(index)] = {m["id"]: np.asarray(m["corners"], dtype=np.float64) for m in definition["markers"]}
    detector = AprilTagDetectorWrapper(tag_family_name=config["tag_family"], refine_edges=True, nthreads=4)
    cameras = [Camera(c, main=(rank == 0)) for rank, c in enumerate(camera_configs)]
    socket = zmq.Context().socket(zmq.PUB)
    socket.bind(f"tcp://*:{config['port']}")
    print(f"publishing on tcp://*:{config['port']}; cameras in priority order: {[c.name for c in cameras]}")

    period = 1.0 / config.get("frequency", 30)
    counts, last_report = {}, time.monotonic()
    try:
        while True:
            start = time.monotonic()
            for camera in cameras[1:]:
                camera.poll()
            if not cameras[0].poll():
                time.sleep(0.002)
                continue
            poses = {}
            for rank, camera in enumerate(cameras):
                missing = [b for b in boards if b not in poses]
                if not missing:
                    break
                if rank > 0 and (camera.frame is None or start - camera.stamp > STALE_SECONDARY_S):
                    continue
                poses.update(board_poses(camera, camera.gray(), detector, boards, missing))
            if not args.headless and show(cameras, poses):
                break
            draw = {b: p.pop("_draw") for b, p in poses.items()}
            if poses:
                socket.send_json({"timestamp": time.time(), "poses": poses})
            for b, p in poses.items():
                counts[(b, p["camera"])] = counts.get((b, p["camera"]), 0) + 1
            if start - last_report > 2.0:
                print("board poses in the last 2 s by camera:",
                      {f"board {b} from {c}": n for (b, c), n in sorted(counts.items())})
                counts, last_report = {}, start
            time.sleep(max(0.0, period - (time.monotonic() - start)))
    finally:
        for camera in cameras:
            camera.release()
        cv2.destroyAllWindows()


def show(cameras, poses):
    for camera in cameras:
        if camera.frame is None:
            continue
        image = camera.frame.copy()
        for board, pose in poses.items():
            if pose["camera"] == camera.name:
                rvec, tvec = pose["_draw"]
                cv2.drawFrameAxes(image, camera.device.camera_matrix, camera.device.dist_coeffs, rvec, tvec, 0.05, 3)
                cv2.putText(image, f"board {board}", (20, 40 + 30 * int(board)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2)
        cv2.imshow(camera.name, cv2.resize(image, None, fx=0.6, fy=0.6))
    return cv2.waitKey(1) & 0xFF == ord("q")


if __name__ == "__main__":
    main()
