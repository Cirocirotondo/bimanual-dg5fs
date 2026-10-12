"""The two Limbs as the training environment uses them.

Everything robot-specific comes from the Scene (bimanual_dg5fs/scene.py) and
the Limb URDFs (docs/adr/0001), so training and teleoperation agree on base
poses, joint order and drive gains.
"""
import math

import torch

from bimanual_dg5fs import scene
from bimanual_training import ASSETS_DIR, USD_DIR

SIDES = scene.SIDES                                   # ("left", "right")
ARM_JOINTS = scene.ARM_JOINTS
HAND_JOINTS = scene.HAND_JOINTS
JOINTS = ARM_JOINTS + HAND_JOINTS                     # per-Limb order, 6 + 20
NUM_ARM, NUM_HAND = len(ARM_JOINTS), len(HAND_JOINTS)
NUM_JOINTS = NUM_ARM + NUM_HAND

PALM_LINK = "link_base"                               # the Hand base (CONTEXT.md)
WRIST_LINK = "wrist_3_link"                           # Mount, Hand base merged into it in the USD
FINGERTIP_LINKS = tuple(f"link_{f}_tip" for f in range(1, 6))
DISTAL_LINKS = tuple(f"link_{f}_4" for f in range(1, 6))   # tips merged into these


def urdf_path(side):
    return ASSETS_DIR / f"ur5e_{side}_dg5fs.urdf"


def usd_path(side):
    # UrdfConverter writes {usd_dir}/{robot}/{robot}.usda; usd_dir is per Limb so the
    # two conversions keep separate config.yaml / hash files.
    name = f"ur5e_{side}_dg5fs"
    return USD_DIR / name / name / f"{name}.usda"


def base_pose(side):
    """World pose of the Limb's root ``base_link``: position and (x, y, z, w) quaternion.

    The Scene's Base poses are poses of the URDF ``base`` link; ``base_link`` is
    turned by pi about z from it (bimanual_dg5fs/scene.py).
    """
    yaw = math.radians(scene.BASE_YAW_DEG[side] - scene.URDF_BASE_LINK_TO_BASE_YAW_DEG)
    return tuple(scene.BASE_POS[side]), (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def base_rotation(side, device="cpu", dtype=torch.float32):
    """``(3, 3)`` rotation of ``base_link`` in the world."""
    yaw = math.radians(scene.BASE_YAW_DEG[side] - scene.URDF_BASE_LINK_TO_BASE_YAW_DEG)
    c, s = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)


def drive_gains():
    """Per-joint (stiffness, damping, effort limit, armature): the Scene's position actuators,
    i.e. the drives the Demonstrations were recorded with."""
    stiffness = dict(zip(ARM_JOINTS, scene.ARM_KP))
    damping = dict(zip(ARM_JOINTS, scene.ARM_KV))
    effort = dict(zip(ARM_JOINTS, scene.ARM_FORCE))
    armature = {j: scene.ARM_ARMATURE for j in ARM_JOINTS}
    for j in HAND_JOINTS:
        stiffness[j], damping[j], effort[j], armature[j] = (scene.HAND_KP, scene.HAND_KV,
                                                            scene.HAND_FORCE, scene.HAND_ARMATURE)
    return stiffness, damping, effort, armature
