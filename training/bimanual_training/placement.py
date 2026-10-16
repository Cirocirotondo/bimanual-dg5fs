"""Object placements (CONTEXT.md): planar moves of the Demonstration's start pose.

A placement is a translation (x, y) along the Table plus a yaw about the vertical
axis through the Object's centre at the Reference's first frame (the pivot), so
translation and yaw are independent knobs. The whole Reference moves rigidly
with it: every position p and orientation q of the Reference becomes

    p' = R_z(yaw) (p - pivot) + pivot + t        q' = q_yaw * q

The cube looks the same every 90 degrees, so yaws are folded into [-45, +45].
Plain torch, robot-agnostic (from simtoolreal_newton's transform_bank / recovery).
"""
import math

import torch

from bimanual_training.rotations import quat_multiply, quat_rotate

YAW_PERIOD_RAD = 0.5 * math.pi        # the cube's symmetry about the vertical
# Exchange rate between yaw and translation when looking for the nearest
# placement (simtoolreal_newton's nearest_yaw_lever_arm_m): 10 deg ~ 1.75 cm.
NEAREST_YAW_LEVER_ARM_M = 0.10


def fold_yaw(yaw):
    """Yaw [rad] folded into [-45, +45] degrees (the cube's 90-degree symmetry)."""
    yaw = torch.as_tensor(yaw)
    return torch.remainder(yaw + 0.5 * YAW_PERIOD_RAD, YAW_PERIOD_RAD) - 0.5 * YAW_PERIOD_RAD


def yaw_quaternion(yaw):
    """``(..., 4)`` xyzw rotations about z."""
    yaw = torch.as_tensor(yaw)
    half = 0.5 * yaw
    zero = torch.zeros_like(half)
    return torch.stack((zero, zero, torch.sin(half), torch.cos(half)), dim=-1)


def yaw_matrix(yaw):
    """``(..., 3, 3)`` rotations about z."""
    yaw = torch.as_tensor(yaw)
    c, s = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((torch.stack((c, -s, zero), -1), torch.stack((s, c, zero), -1),
                        torch.stack((zero, zero, one), -1)), -2)


def transform_points(points, translation, yaw, pivot):
    """Move ``points`` (..., 3) by placements broadcasting against them.

    ``translation`` (..., 3), ``yaw`` (...,), ``pivot`` (3,).
    """
    rotation = yaw_matrix(yaw)
    relative = points - pivot
    rotated = (rotation @ relative.unsqueeze(-1)).squeeze(-1)
    return rotated + pivot + translation


def transform_quaternions(quaternions, yaw):
    """``q' = q_yaw * q`` for xyzw quaternions (..., 4)."""
    return quat_multiply(yaw_quaternion(yaw).expand_as(quaternions), quaternions)


def transform_vectors(vectors, yaw):
    """Rotate free vectors (velocities) by the placement's yaw."""
    return quat_rotate(yaw_quaternion(yaw).expand(*vectors.shape[:-1], 4), vectors)


def placement_distance(translation_a, yaw_a, translation_b, yaw_b,
                       lever_arm_m=NEAREST_YAW_LEVER_ARM_M):
    """Squared distance between placements: |dxy|^2 + (lever arm * folded dyaw)^2."""
    dxy = (translation_a[..., :2] - translation_b[..., :2]).square().sum(-1)
    dyaw = fold_yaw(yaw_a - yaw_b)
    return dxy + (lever_arm_m * dyaw).square()


def nearest_placements(translation, yaw, bank_translation, bank_yaw):
    """Index of the nearest bank placement for each query ``(N, 3)``, ``(N,)``."""
    distance = placement_distance(translation.unsqueeze(1), yaw.unsqueeze(1),
                                  bank_translation.unsqueeze(0), bank_yaw.unsqueeze(0))
    return distance.argmin(dim=1)
