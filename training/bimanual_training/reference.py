"""A Reference (CONTEXT.md) loaded on the training device.

Built by scripts/build_reference.py. Every Limb tensor is ``(T, 2, ...)``, the
Limb axis in ``limbs.SIDES`` order; positions and orientations are in the world
(= env) frame, quaternions xyzw. An articulated Object (the book) also has its
joint positions (``object_joint_q``, ``(T, J)`` in the Scene's OBJECT_JOINTS order);
a rigid one has ``J = 0``. A Scene with more than one free body (bimanual_dg5fs.scene
EXTRA_OBJECTS: the cube beside the basin) also has each extra body's track, read as
``<name>_pose`` ``(T, 7)`` and ``<name>_velocity`` ``(T, 6)`` like the Object's.
"""
import json
from dataclasses import dataclass, field

import numpy as np
import torch

from bimanual_training import REFERENCES_DIR, ROOT_DIR, limbs
from bimanual_training.rotations import quat_rotate

PALM_LEVER_ARM_M = 0.1     # keypoints.py: the exchange rate between palm metres and radians
DEMONSTRATIONS_DIR = ROOT_DIR / "teleoperation" / "demonstrations"


def reference_object(name):
    """The Object (bimanual_dg5fs.scene.OBJECTS) the Reference ``name`` was demonstrated with:
    its sidecar's ``object``, else its Demonstration's sidecar ``scene.object``, else the cube
    (the References built before the basin)."""
    sidecar = REFERENCES_DIR / f"{name}.json"
    info = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    if "object" in info:
        return info["object"]
    demonstration = DEMONSTRATIONS_DIR / info.get("demonstration", "")
    if info.get("demonstration") and demonstration.with_suffix(".json").is_file():
        return json.loads(demonstration.with_suffix(".json").read_text()).get("scene", {}).get("object", "cube")
    return "cube"


def palm_keypoints(position, quat, lever_arm_m=PALM_LEVER_ARM_M):
    """``(..., 4, 3)``: the Hand-base origin and three points at ``lever_arm_m`` along its axes."""
    axes = torch.eye(3, dtype=position.dtype, device=position.device) * lever_arm_m
    axes = axes.expand(*position.shape[:-1], 3, 3)
    points = position.unsqueeze(-2) + quat_rotate(quat.unsqueeze(-2).expand(*quat.shape[:-1], 3, 4), axes)
    return torch.cat((position.unsqueeze(-2), points), dim=-2)


def object_velocity(pose, dt):
    """``(T, 6)`` world-frame linear and angular velocity of an ``(T, 7)`` xyz + xyzw track."""
    from scipy.spatial.transform import Rotation

    linear = np.gradient(pose[:, :3], dt, axis=0)
    rotations = Rotation.from_quat(pose[:, 3:7])
    step = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt        # world frame
    angular = np.zeros((len(pose), 3))
    if len(pose) > 1:
        angular[1:-1] = 0.5 * (step[1:] + step[:-1])
        angular[0], angular[-1] = step[0], step[-1]
    return np.concatenate((linear, angular), axis=1)


def joint_velocity(q, dt):
    """``(T, J)`` velocity of a ``(T, J)`` joint track by central differences."""
    if q.shape[1] == 0 or len(q) < 2:
        return np.zeros_like(q)
    return np.gradient(q, dt, axis=0)


@dataclass
class Reference:
    dt: float
    q: torch.Tensor                  # (T, 2, 26) ARM_JOINTS + HAND_JOINTS
    dq: torch.Tensor                 # (T, 2, 26)
    palm_position: torch.Tensor      # (T, 2, 3)
    palm_quat: torch.Tensor          # (T, 2, 4)
    fingertips: torch.Tensor         # (T, 2, 5, 3)
    palm_keypoints: torch.Tensor     # (T, 2, 4, 3)
    object_pose: torch.Tensor        # (T, 7) x y z qx qy qz qw, for the later Training stages
    object_velocity: torch.Tensor    # (T, 6) linear [m/s], angular [rad/s], world frame
    object_joint_q: torch.Tensor     # (T, J) an articulated Object's joints [rad]; J = 0 for a rigid one
    object_joint_dq: torch.Tensor    # (T, J) [rad/s]
    # The Scene's other free bodies (EXTRA_OBJECTS), by name: (T, 7) poses and (T, 6) velocities
    # as for the Object; empty for a Scene with the Object only.
    extra_pose: dict[str, torch.Tensor] = field(default_factory=dict)
    extra_velocity: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getattr__(self, key):
        # "<name>_pose" / "<name>_velocity" of an extra body, so ref("cube_pose", ...) reads like the Object's.
        extras = self.__dict__.get("extra_pose", {})
        for suffix, table in (("_pose", "extra_pose"), ("_velocity", "extra_velocity")):
            if key.endswith(suffix) and key[:-len(suffix)] in extras:
                return self.__dict__[table][key[:-len(suffix)]]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {key!r}")

    @property
    def extra_objects(self):
        """The extra free bodies' names (bimanual_dg5fs.scene.EXTRA_OBJECTS), sorted."""
        return tuple(sorted(self.extra_pose))

    @property
    def object_joints(self):
        return self.object_joint_q.shape[1]

    @property
    def frames(self):
        return self.q.shape[0]

    @property
    def last_index(self):
        return self.frames - 1

    @classmethod
    def load(cls, name, device="cpu"):
        path = REFERENCES_DIR / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"No Reference {path}; build it with training/scripts/build_reference.py")
        data = np.load(path)

        def stacked(key):
            return torch.as_tensor(np.stack([data[f"{side}_{key}"] for side in limbs.SIDES], axis=1),
                                   dtype=torch.float32, device=device)

        palm_position, palm_quat = stacked("palm_position"), stacked("palm_quat")
        object_pose = np.asarray(data["object_pose"], dtype=np.float64)
        dt = float(data["dt"])
        joint_q = (np.asarray(data["object_joint_q"], dtype=np.float64) if "object_joint_q" in data.files
                   else np.zeros((len(object_pose), 0)))

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        # The extra free bodies, stored as "<name>_pose" (scripts/build_reference.py).
        extra = {key[:-len("_pose")]: np.asarray(data[key], dtype=np.float64)
                 for key in data.files if key.endswith("_pose") and key != "object_pose"}

        return cls(
            dt=float(data["dt"]),
            q=stacked("q"),
            dq=stacked("dq"),
            palm_position=palm_position,
            palm_quat=palm_quat,
            fingertips=stacked("fingertips"),
            palm_keypoints=palm_keypoints(palm_position, palm_quat),
            object_pose=torch.as_tensor(object_pose, dtype=torch.float32, device=device),
            object_velocity=tensor(object_velocity(object_pose, dt)),
            object_joint_q=tensor(joint_q),
            object_joint_dq=tensor(joint_velocity(joint_q, dt)),
            extra_pose={name: tensor(pose) for name, pose in extra.items()},
            extra_velocity={name: tensor(object_velocity(pose, dt)) for name, pose in extra.items()},
        )
