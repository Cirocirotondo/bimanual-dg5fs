"""Generalised imitation (Training stage 3, CONTEXT.md): the Object anywhere a
Transform bank covers.

Each episode picks a bank entry (an Object placement) with the configured mix,
and the whole Reference moves rigidly with it: the Hand-base targets, fingertips
and Object track are the Reference's moved by the placement, the Arms' joints
come from the bank's IK (for Reference state initialization), the Hands' are the
Demonstration's. Rewards and observations keep stage 2's: the Hand rewards
compare the palms with the moved Reference, i.e. relative to where the Object
should be, and the Object rewards compare the Object with its moved track.

Canonical frame (option): the policy observes the Hand-base poses moved back by
the inverse placement, the Arms' joints as their deviation from the moved
Reference added to the Demonstration's, and its twists are rotated by the
placement's yaw, so a followed Reference looks like the Demonstration's own.

Object symmetry: wherever the Object's orientation is used (rewards,
observations), it is the one of its 24 symmetry-equivalent orientations closest
to the Reference's current orientation.
"""
from __future__ import annotations

import math

import torch

from bimanual_training import TRAINING_DIR, limbs
from bimanual_training.cuboid_symmetry import cuboid_rotation_symmetries
from bimanual_training.motion_imitation_cfg import OBSERVATIONS_PER_LIMB, BimanualPlacementImitationEnvCfg
from bimanual_training.object_imitation_env import BimanualObjectImitationEnv
from bimanual_training.placement import fold_yaw, transform_points, transform_quaternions, transform_vectors
from bimanual_training.rotations import normalize_canonical_quaternion, quat_multiply

NA = limbs.NUM_ARM


class BimanualPlacementImitationEnv(BimanualObjectImitationEnv):
    cfg: BimanualPlacementImitationEnvCfg

    def __init__(self, cfg: BimanualPlacementImitationEnvCfg, render_mode: str | None = None, **kwargs):
        device = str(cfg.sim.device)
        bank = torch.load(TRAINING_DIR / "banks" / f"{cfg.bank}.pt", map_location=device, weights_only=True)
        self.bank_translation = bank["translation"].to(device)                   # (T, 3)
        self.bank_yaw = bank["yaw"].to(device)                                   # (T,)
        self.bank_arm_q = bank["arm_q"].to(device)                               # (T, F, 2, 6)
        if cfg.observe_placement:
            cfg.observation_space = 2 * OBSERVATIONS_PER_LIMB + 1 + 2 * 9 + 3
        super().__init__(cfg, render_mode, **kwargs)
        if cfg.object != "cube":
            raise ValueError(f"Generalised imitation assumes the cube (its Object symmetries), not the {cfg.object}")
        if self.bank_arm_q.shape[1] != self.reference.frames:
            raise ValueError(f"bank {cfg.bank} has {self.bank_arm_q.shape[1]} frames, "
                             f"the Reference {self.reference.frames}")
        self.pivot = self.reference.object_pose[0, :3].clone()
        n = self.num_envs
        self.placement = torch.zeros(n, dtype=torch.long, device=self.device)
        self.symmetries = cuboid_rotation_symmetries((cfg.object_size_m / 2.0,) * 3).to(self.device, torch.float32)
        self.placement_probability = self._placement_mix()

    def _placement_mix(self):
        """Probability of each bank entry: demo_share on entry 0 (the Demonstration's
        placement), the near band weighted near_band_weight, the rest uniform."""
        cfg = self.cfg
        near = ((self.bank_translation[:, :2].norm(dim=-1) <= cfg.near_band_m)
                & (fold_yaw(self.bank_yaw).abs() <= math.radians(cfg.near_band_deg)))
        weight = torch.where(near, torch.full_like(self.bank_yaw, cfg.near_band_weight), torch.ones_like(self.bank_yaw))
        weight[0] = 0.0
        probability = weight / weight.sum() * (1.0 - cfg.demo_share)
        probability[0] = cfg.demo_share
        return probability

    # -------------------------------------------------------------- placement

    def _before_reset(self, env_ids):
        if self.cfg.placement_index is not None:
            self.placement[env_ids] = int(self.cfg.placement_index)
        else:
            self.placement[env_ids] = torch.multinomial(self.placement_probability, len(env_ids), replacement=True)

    def ref(self, key, index, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        placement = self.placement[ids]
        translation, yaw = self.bank_translation[placement], self.bank_yaw[placement]
        if key == "q":
            arm = self.bank_arm_q[placement, index]                                              # (N, 2, 6)
            return torch.cat((arm, self.reference.q[index][:, :, NA:]), dim=-1)
        if key == "dq":
            last = self.reference.last_index
            before, after = (index - 1).clamp(min=0), (index + 1).clamp(max=last)
            span = ((after - before).clamp(min=1).float() * self.reference.dt).view(-1, 1, 1)
            arm = (self.bank_arm_q[placement, after] - self.bank_arm_q[placement, before]) / span
            return torch.cat((arm, self.reference.dq[index][:, :, NA:]), dim=-1)
        if key in ("palm_position", "palm_keypoints", "fingertips"):
            points = getattr(self.reference, key)[index]                                         # (N, 2, ..., 3)
            expand = (-1,) + (1,) * (points.dim() - 2)
            return transform_points(points, translation.view(*expand, 3), yaw.view(*expand), self.pivot)
        if key == "palm_quat":
            return transform_quaternions(self.reference.palm_quat[index], yaw.view(-1, 1))
        if key == "object_pose":
            pose = self.reference.object_pose[index]
            return torch.cat((transform_points(pose[:, :3], translation, yaw, self.pivot),
                              transform_quaternions(pose[:, 3:7], yaw)), dim=-1)
        if key == "object_velocity":
            velocity = self.reference.object_velocity[index]
            return torch.cat((transform_vectors(velocity[:, :3], yaw), transform_vectors(velocity[:, 3:], yaw)), dim=-1)
        return super().ref(key, index, env_ids)

    def _extra_observations(self):
        observation = super()._extra_observations()
        if not self.cfg.observe_placement:
            return observation
        placement = torch.cat((self.bank_translation[self.placement, :2],
                               fold_yaw(self.bank_yaw[self.placement]).unsqueeze(1)), dim=1)
        return torch.cat((observation, placement), dim=1)

    # ---------------------------------------------------------- canonical frame

    def _observed_palm_pose(self, position, quat):
        if not self.cfg.canonical_frame:
            return position, quat
        translation, yaw = self.bank_translation[self.placement], self.bank_yaw[self.placement]
        position = transform_points(position - translation.unsqueeze(1), torch.zeros_like(translation).unsqueeze(1),
                                    -yaw.view(-1, 1), self.pivot)
        return position, normalize_canonical_quaternion(transform_quaternions(quat, -yaw.view(-1, 1)))

    def _observed_joints(self, q, targets, dq):
        if not self.cfg.canonical_frame:
            return q, targets, dq
        # The Arms' deviation from the moved Reference, carried onto the Demonstration's joints.
        index = self.reference_index
        offset = torch.zeros_like(q)
        offset[..., :NA] = self.reference.q[index][..., :NA] - self.ref("q", index)[..., :NA]
        velocity_offset = torch.zeros_like(dq)
        velocity_offset[..., :NA] = self.reference.dq[index][..., :NA] - self.ref("dq", index)[..., :NA]
        return q + offset, targets + offset, dq + velocity_offset

    def _world_twist(self, twist_actions):
        if not self.cfg.canonical_frame:
            return twist_actions
        yaw = self.bank_yaw[self.placement].view(-1, 1)
        return torch.cat((transform_vectors(twist_actions[..., 0:3], yaw),
                          transform_vectors(twist_actions[..., 3:6], yaw), twist_actions[..., 6:]), dim=-1)

    def reference_actions(self):
        actions = super().reference_actions()
        if not self.cfg.canonical_frame:
            return actions
        per_limb = actions.view(self.num_envs, 2, -1)
        yaw = -self.bank_yaw[self.placement].view(-1, 1)
        per_limb[..., 0:3] = transform_vectors(per_limb[..., 0:3], yaw)
        per_limb[..., 3:6] = transform_vectors(per_limb[..., 3:6], yaw)
        return per_limb.view(self.num_envs, -1)

    # ----------------------------------------------------------- symmetry

    def object_pose(self):
        """The Object's pose, its orientation the symmetry-equivalent one closest to the
        Reference's current orientation (Object symmetry)."""
        position, quat = super().object_pose()
        reference = normalize_canonical_quaternion(self.ref("object_pose", self.reference_index)[:, 3:7])
        candidates = quat_multiply(quat.unsqueeze(1).expand(-1, len(self.symmetries), -1),
                                   self.symmetries.unsqueeze(0).expand(len(quat), -1, -1))         # (N, 24, 4)
        best = (candidates * reference.unsqueeze(1)).sum(-1).abs().argmax(dim=1)
        chosen = candidates[torch.arange(len(quat), device=self.device), best]
        return position, normalize_canonical_quaternion(chosen)
