"""From Wrist board poses to Hand-base targets, one Limb at a time.

Ported from the UR5 PC arm teleoperation (deployment/teleoperation:
pose_stream.WristPoseFilter, transforms.RelativeBoardMapper in the
"spatial-relative" mode, and its target rate limits), with two changes:

* the Wrist calibration is applied in full metres (the UR5 config divides it
  by 100 by mistake), so the robot Hand base follows the Operator's wrist and
  not the board;
* everything is expressed in the Scene world frame. The Operator stands behind
  the Arm bases facing -y, so World board axes and Scene axes coincide
  (``axis_map`` = identity) for both Limbs.
"""
import json
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Pose:
    position: np.ndarray        # (3,)
    rotation: np.ndarray        # (3, 3)

    def copy(self):
        return Pose(self.position.copy(), self.rotation.copy())


def rotation_angle_deg(rot_a, rot_b):
    return float(np.degrees(R.from_matrix(rot_a.T @ rot_b).magnitude()))


class WristPoseFilter:
    """Rejects tracking jumps, then low-pass filters position and rotation.

    The pose estimator occasionally reports a single wild pose. If the filter
    happened to start on such a sample it would reject every good sample after
    it, so after ``reacquire_after`` consecutive rejected samples that agree
    with each other it restarts from the latest one.
    """

    def __init__(self, translation_alpha, rotation_alpha, max_jump_m, max_jump_deg, reacquire_after=5):
        self.translation_alpha = translation_alpha
        self.rotation_alpha = rotation_alpha
        self.max_jump_m = max_jump_m
        self.max_jump_deg = max_jump_deg
        self.reacquire_after = reacquire_after
        self.pose = None
        self.rejected = []

    def reset(self):
        self.pose = None
        self.rejected = []

    def _close(self, a, b):
        return (np.linalg.norm(a.position - b.position) <= self.max_jump_m
                and rotation_angle_deg(a.rotation, b.rotation) <= self.max_jump_deg)

    def update(self, pose):
        """Returns the filtered pose, or None if the sample was rejected as a jump."""
        if self.pose is None:
            self.pose = pose.copy()
            return self.pose.copy()
        if not self._close(self.pose, pose):
            if self.rejected and not self._close(self.rejected[-1], pose):
                self.rejected = []
            self.rejected.append(pose.copy())
            if len(self.rejected) >= self.reacquire_after:
                self.pose, self.rejected = pose.copy(), []
                return self.pose.copy()
            return None
        self.rejected = []
        a = self.translation_alpha
        position = (1.0 - a) * self.pose.position + a * pose.position
        delta = R.from_matrix(self.pose.rotation.T @ pose.rotation).as_rotvec()
        rotation = self.pose.rotation @ R.from_rotvec(self.rotation_alpha * delta).as_matrix()
        self.pose = Pose(position, rotation)
        return self.pose.copy()


class WristCalibration:
    """Wrist board -> Operator's wrist. Identity until a Glove is calibrated."""

    def __init__(self, translation=(0.0, 0.0, 0.0), rotation=None):
        self.translation = np.asarray(translation, dtype=float)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)

    @classmethod
    def from_file(cls, path):
        """Reads ``T_board_from_wrist`` (wrist frame -> board frame), as written by
        the robohand Glove calibration (right_wrist2board_calibration.json)."""
        data = json.loads(open(path).read())
        matrix = np.asarray(data["T_board_from_wrist"]["matrix_4x4"], dtype=float)
        return cls(matrix[:3, 3], matrix[:3, :3])

    def wrist_pose(self, board_pose):
        return Pose(board_pose.position + board_pose.rotation @ self.translation,
                    board_pose.rotation @ self.rotation)


class RelativeMapper:
    """Spatial-relative mapping from the Operator's wrist to the Hand base.

    At engagement the wrist pose and the Hand-base pose form the Neutral pose.
    Afterwards
        p_target = p_hand0 + scale * A (p_wrist - p_wrist0)
        R_target = exp(A rotvec(R_wrist R_wrist0^T)) R_hand0
    i.e. the Hand base translates and rotates (about itself) by the wrist's
    motion expressed in world axes.
    """

    def __init__(self, axis_map=np.eye(3), scale=1.0):
        self.axis_map = np.asarray(axis_map, dtype=float)
        self.scale = scale
        self.wrist0 = None
        self.hand0 = None

    def engage(self, wrist_pose, hand_pose):
        self.wrist0 = wrist_pose.copy()
        self.hand0 = hand_pose.copy()

    def target(self, wrist_pose):
        dp = wrist_pose.position - self.wrist0.position
        rotvec = R.from_matrix(wrist_pose.rotation @ self.wrist0.rotation.T).as_rotvec()
        position = self.hand0.position + self.scale * self.axis_map @ dp
        rotation = R.from_rotvec(self.axis_map @ rotvec).as_matrix() @ self.hand0.rotation
        return Pose(position, rotation)


def rate_limit(previous, target, dt, max_speed_m_s, max_rotation_speed_deg_s):
    """Moves ``previous`` towards ``target`` by at most the allowed step."""
    step = target.position - previous.position
    max_step = max_speed_m_s * dt
    norm = np.linalg.norm(step)
    position = previous.position + (step * (max_step / norm) if norm > max_step else step)

    delta = R.from_matrix(previous.rotation.T @ target.rotation).as_rotvec()
    max_angle = np.radians(max_rotation_speed_deg_s) * dt
    angle = np.linalg.norm(delta)
    if angle > max_angle:
        delta = delta * (max_angle / angle)
    rotation = previous.rotation @ R.from_rotvec(delta).as_matrix()
    return Pose(position, rotation)
