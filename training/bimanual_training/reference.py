"""A Reference (CONTEXT.md) loaded on the training device.

Built by scripts/build_reference.py. Every tensor is ``(T, 2, ...)``, the Limb
axis in ``limbs.SIDES`` order; positions and orientations are in the world
(= env) frame, quaternions xyzw.
"""
from dataclasses import dataclass

import numpy as np
import torch

from bimanual_training import REFERENCES_DIR, limbs
from bimanual_training.rotations import quat_rotate

PALM_LEVER_ARM_M = 0.1     # keypoints.py: the exchange rate between palm metres and radians


def palm_keypoints(position, quat, lever_arm_m=PALM_LEVER_ARM_M):
    """``(..., 4, 3)``: the Hand-base origin and three points at ``lever_arm_m`` along its axes."""
    axes = torch.eye(3, dtype=position.dtype, device=position.device) * lever_arm_m
    axes = axes.expand(*position.shape[:-1], 3, 3)
    points = position.unsqueeze(-2) + quat_rotate(quat.unsqueeze(-2).expand(*quat.shape[:-1], 3, 4), axes)
    return torch.cat((position.unsqueeze(-2), points), dim=-2)


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
        return cls(
            dt=float(data["dt"]),
            q=stacked("q"),
            dq=stacked("dq"),
            palm_position=palm_position,
            palm_quat=palm_quat,
            fingertips=stacked("fingertips"),
            palm_keypoints=palm_keypoints(palm_position, palm_quat),
            object_pose=torch.as_tensor(data["object_pose"], dtype=torch.float32, device=device),
        )
