"""Demonstrations: one teleoperation episode in the Scene, saved as .npz + .json.

One sample per control tick. Poses are [x, y, z, qx, qy, qz, qw] in the Scene
world frame. The raw inputs (Wrist board poses in the World board frame and
Glove ergonomics angles) are kept so a Demonstration can be replayed through
the whole pipeline (``--source replay``).

npz keys, per side (``right_`` / ``left_`` prefix):
  arm_q_commanded (T,6), arm_q_measured (T,6), hand_q_commanded (T,20),
  hand_q_measured (T,20), hand_base_target (T,7), hand_base_measured (T,7),
  engaged (T,), wrist_position (T,3), wrist_rotation (T,3,3),
  wrist_confidence (T,), wrist_valid (T,), wrist_alt_position (T,3), wrist_alt_rotation (T,3,3),
  wrist_alt_valid (T,) (the other pose of an ambiguous single tag), wrist_used_alternative (T,)
  (the Glove IMU chose it), imu_quaternion (T,4) xyzw, imu_valid (T,),
  glove_ergonomics_deg (T,20), glove_valid (T,)
shared: time (T,) monotonic seconds, object_pose (T,7), and for an articulated
  Object object_joint_q (T,k) in the order of the JSON's scene.object_joints, and
  <name>_pose (T,7) for each of the Scene's other free bodies (JSON scene.extra_objects)
"""
import datetime
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from teleoperation.inputs import SIDES

SCHEMA_VERSION = 2


def _pose7(pose):
    return np.r_[pose.position, R.from_matrix(pose.rotation).as_quat()]


class DemonstrationRecorder:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.metadata = metadata
        self.samples = None
        self.started = None

    @property
    def active(self):
        return self.samples is not None

    def start(self):
        self.samples = []
        self.started = datetime.datetime.now(datetime.timezone.utc)

    def discard(self):
        self.samples = None

    def add(self, now, inputs, targets, measured):
        if self.samples is None:
            return
        sample = {"time": now, "object_pose": measured["object"]}
        if len(measured.get("object_joints", ())):
            sample["object_joint_q"] = measured["object_joints"]
        for name, pose in measured.get("extra_objects", {}).items():
            sample[f"{name}_pose"] = pose
        for side in SIDES:
            t, m = targets[side], measured[side]
            wrist, glove, imu = inputs.wrist[side], inputs.glove[side], inputs.imu[side]
            alt = wrist.alternatives[0] if wrist is not None and wrist.alternatives else None
            sample.update({
                f"{side}_arm_q_commanded": t.arm_q,
                f"{side}_arm_q_measured": m["arm_q"],
                f"{side}_hand_q_commanded": t.hand_q,
                f"{side}_hand_q_measured": m["hand_q"],
                f"{side}_hand_base_target": _pose7(t.hand_base),
                f"{side}_hand_base_measured": m["hand_base"],
                f"{side}_engaged": t.engaged,
                f"{side}_wrist_position": wrist.pose.position if wrist else np.zeros(3),
                f"{side}_wrist_rotation": wrist.pose.rotation if wrist else np.eye(3),
                f"{side}_wrist_confidence": wrist.confidence if wrist else 0.0,
                f"{side}_wrist_valid": wrist is not None,
                f"{side}_wrist_alt_position": alt.position if alt else np.zeros(3),
                f"{side}_wrist_alt_rotation": alt.rotation if alt else np.eye(3),
                f"{side}_wrist_alt_valid": alt is not None,
                f"{side}_wrist_used_alternative": t.used_alternative,
                f"{side}_imu_quaternion": imu.rotation.as_quat() if imu else np.array([0.0, 0.0, 0.0, 1.0]),
                f"{side}_imu_valid": imu is not None,
                f"{side}_glove_ergonomics_deg": glove.ergonomics_deg if glove else np.zeros(20),
                f"{side}_glove_valid": glove is not None,
            })
        self.samples.append(sample)

    def save(self, success, termination):
        """Writes the episode and stops recording. Returns the .npz path (None if empty)."""
        samples, self.samples = self.samples, None
        if not samples:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = "demo_" + self.started.strftime("%Y%m%d_%H%M%S_%f")
        arrays = {key: np.asarray([s[key] for s in samples]) for key in samples[0]}
        np.savez_compressed(self.directory / f"{stem}.npz", **arrays)
        info = {
            "schema_version": SCHEMA_VERSION,
            "started_utc": self.started.isoformat(),
            "samples": len(samples),
            "duration_s": float(arrays["time"][-1] - arrays["time"][0]),
            "success": success,
            "termination": termination,
            **self.metadata,
        }
        (self.directory / f"{stem}.json").write_text(json.dumps(info, indent=2))
        return self.directory / f"{stem}.npz"
