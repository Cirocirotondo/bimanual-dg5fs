"""Motion imitation of a Reference with both Limbs (Training stage 1, CONTEXT.md).

simtoolreal_newton's motion-imitation environment cut down to what this stage
needs and widened to two Limbs:

* each Limb is an Articulation from its own URDF, placed at the Scene's Base
  pose; the two Limbs collide with each other, not with themselves;
* actions, per Limb: a Hand-base twist in the world frame turned into arm joint
  targets by one damped-least-squares IK step, and hand joint targets as
  residuals on a default pose (52 in all, ``limbs.SIDES`` order);
* Reference state initialization at a uniform frame, one control step per
  Reference frame, the episode ends with the Reference;
* rewards on Hand-base keypoints in the world, hand joints and action rates;
  early termination when a Limb strays past a threshold for a few steps.

Per-Limb tensors carry the Limb axis second: ``(num_envs, 2, ...)``.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from bimanual_training import limbs
from bimanual_training.kinematics import LimbKinematics
from bimanual_training.motion_imitation_cfg import ACTIONS_PER_LIMB, BimanualMotionImitationEnvCfg, make_limb_cfg
from bimanual_training.operational_space import damped_least_squares_step, saturate_direction_preserving
from bimanual_training.reference import PALM_LEVER_ARM_M, Reference, palm_keypoints
from bimanual_training.rotations import (
    matrix_to_quat,
    normalize_canonical_quaternion,
    quat_multiply,
    quat_rotate,
    quat_rotate_inverse,
    quat_to_rotation_6d,
)

NUM_LIMBS = len(limbs.SIDES)
NA = limbs.NUM_ARM


def gaussian(mean_squared_error, std):
    return torch.exp(-mean_squared_error / (2.0 * float(std) ** 2))


class BimanualMotionImitationEnv(DirectRLEnv):
    cfg: BimanualMotionImitationEnvCfg

    def __init__(self, cfg: BimanualMotionImitationEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.left_limb is None:
            cfg.left_limb = make_limb_cfg("left")
        if cfg.right_limb is None:
            cfg.right_limb = make_limb_cfg("right")
        device = str(cfg.sim.device)
        self.reference = Reference.load(cfg.reference, device=device)
        if abs(self.reference.dt - cfg.sim.dt * cfg.decimation) > 1e-6:
            raise ValueError(f"Reference at {self.reference.dt} s per frame, control step {cfg.sim.dt * cfg.decimation}")
        cfg.episode_length_s = self.reference.frames * self.reference.dt
        self.kinematics = [LimbKinematics(limbs.urdf_path(side), limbs.ARM_JOINTS, limbs.HAND_JOINTS, device=device)
                           for side in limbs.SIDES]
        super().__init__(cfg, render_mode, **kwargs)
        self._resolve_layout()
        self._allocate_buffers()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.limbs = [Articulation(getattr(self.cfg, f"{side}_limb")) for side in limbs.SIDES]
        self._spawn_extras()
        positions = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing)[0]
        plan = cloner.clone_plan_from_env_0("/World/envs/env_0", "/World/envs/env_{}", self.scene.num_envs,
                                            positions, global_paths=())
        cloner.replicate(plan)
        for side, limb in zip(limbs.SIDES, self.limbs):
            self.scene.articulations[f"{side}_limb"] = limb
        self._register_extras()
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def ref(self, key, index, env_ids=None):
        """The Reference's ``key`` (q, dq, palm_position, palm_quat, palm_keypoints, fingertips,
        object_pose, object_velocity) at frames ``index`` for envs ``env_ids`` (all when None).

        Every per-frame read of the Reference goes through here, so a Training stage
        that moves the Reference per episode (Object placements) overrides only this.
        """
        return getattr(self.reference, key)[index]

    def _before_reset(self, env_ids):
        """Choose per-episode settings (e.g. the Object placement) before the reset reads the Reference."""

    # Hooks for a Training stage with more than the Limbs (object_imitation_env.py).
    def _spawn_extras(self):
        """Spawn more assets in env_0 before it is cloned."""

    def _register_extras(self):
        """Register the extra assets with the scene after cloning."""

    def _reset_extras(self, env_ids, start):
        """Reset the extra assets of ``env_ids`` to Reference frame ``start``."""

    def _extra_observations(self):
        """``(N, k)`` observations appended after the Phase, or None."""
        return None

    def _observed_palm_pose(self, position, quat):
        """The Hand-base pose as the policy observes it (stage 3 may move it into the
        placement's frame)."""
        return position, quat

    def _observed_joints(self, q, targets, dq):
        """Joints, drive targets and velocities ``(N, 2, 26)`` as the policy observes them."""
        return q, targets, dq

    def _world_twist(self, twist_actions):
        """The policy's twist actions ``(N, 2, 6)`` as world-frame twists."""
        return twist_actions

    def _extra_violation(self):
        """``(N,)`` bool: a stage-specific termination condition this step, or None."""
        return None

    def _resolve_layout(self):
        device = self.device
        self.wrist_body = []
        self.distal_bodies = []
        palm_position, palm_quat, tip_offsets, lower, upper, velocity = [], [], [], [], [], []
        for side, limb, kinematics in zip(limbs.SIDES, self.limbs, self.kinematics):
            if list(limb.joint_names) != list(limbs.JOINTS):
                raise ValueError(f"{side} Limb joints {limb.joint_names} are not in Limb order")
            self.wrist_body.append(limb.find_bodies(limbs.WRIST_LINK)[0][0])
            ids, names = limb.find_bodies(list(limbs.DISTAL_LINKS), preserve_order=True)
            if list(names) != list(limbs.DISTAL_LINKS):
                raise ValueError(f"{side} distal phalanges resolved to {names}")
            self.distal_bodies.append(ids)
            # The Hand base and the fingertips were merged into wrist_3_link and the
            # distal phalanges; their fixed offsets come from the URDF.
            offset = kinematics.offset_from(limbs.WRIST_LINK, limbs.PALM_LINK).to(device)
            palm_position.append(offset[:3, 3])
            palm_quat.append(matrix_to_quat(offset[:3, :3].unsqueeze(0))[0])
            tip_offsets.append(torch.stack([kinematics.offset_from(d, t)[:3, 3]
                                            for d, t in zip(limbs.DISTAL_LINKS, limbs.FINGERTIP_LINKS)]).to(device))
            limits = limb.data.joint_pos_limits.torch[0].clone()
            lower.append(limits[:, 0])
            upper.append(limits[:, 1])
            velocity.append(limb.data.joint_velocity_limits.torch[0].clone())
        self.palm_position_in_wrist = torch.stack(palm_position)              # (2, 3)
        self.palm_quat_in_wrist = torch.stack(palm_quat)                      # (2, 4)
        self.tip_offsets = torch.stack(tip_offsets)                           # (2, 5, 3)
        self.joint_lower = torch.stack(lower)                                 # (2, 26)
        self.joint_upper = torch.stack(upper)
        self.target_slew_per_step = torch.stack(velocity) * self.step_dt
        if not self.cfg.slew_targets_at_velocity_limit:
            self.target_slew_per_step.fill_(float("inf"))
        reference_q = self.reference.q
        outside = torch.maximum(self.joint_lower - reference_q, reference_q - self.joint_upper).amax(dim=0)
        if torch.any(outside > 1e-4):
            worst = [f"{limbs.SIDES[l]}/{limbs.JOINTS[j]} by {float(outside[l, j]):.3f} rad"
                     for l, j in (outside > 1e-4).nonzero().tolist()]
            raise ValueError(f"The Reference leaves the joint limits: {worst}")
        # world <- base_link, per Limb
        self.base_rotation = torch.stack([limbs.base_rotation(side, device=device) for side in limbs.SIDES])
        self.default_hand = self.reference.q[:, :, NA:].mean(dim=0)          # (2, 20)

    def _allocate_buffers(self):
        n, device = self.num_envs, self.device
        self.actions = torch.zeros((n, self.cfg.action_space), device=device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.filtered_actions = torch.zeros_like(self.actions)
        self.position_targets = torch.zeros((n, NUM_LIMBS, limbs.NUM_JOINTS), device=device)
        # The two previous steps' targets, for the targets' acceleration (vibration).
        self.previous_targets = torch.zeros_like(self.position_targets)
        self.previous_targets_2 = torch.zeros_like(self.position_targets)
        self.applied_targets = torch.zeros_like(self.position_targets)
        self.applied_arm_delta = torch.zeros((n, NUM_LIMBS, NA), device=device)
        self.ik_residual = torch.zeros((n, NUM_LIMBS), device=device)
        self.reference_index = torch.zeros(n, dtype=torch.long, device=device)
        self.palm_violation_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.hand_violation_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.extra_violation_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.suppress_arm_action_rate = torch.ones(n, dtype=torch.bool, device=device)
        self.episode_sums = {}

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def q(self):
        return torch.stack([limb.data.joint_pos.torch for limb in self.limbs], dim=1)

    @property
    def dq(self):
        return torch.stack([limb.data.joint_vel.torch for limb in self.limbs], dim=1)

    def palm_pose(self):
        """Hand-base position ``(N, 2, 3)`` in the env frame and orientation ``(N, 2, 4)``."""
        wrist = torch.stack([limb.data.body_link_pose_w.torch[:, body]
                             for limb, body in zip(self.limbs, self.wrist_body)], dim=1)
        wrist_quat = normalize_canonical_quaternion(wrist[..., 3:7])
        offset = self.palm_position_in_wrist.expand_as(wrist[..., 0:3])
        position = wrist[..., 0:3] - self.scene.env_origins.unsqueeze(1) + quat_rotate(wrist_quat, offset)
        quat = normalize_canonical_quaternion(quat_multiply(wrist_quat, self.palm_quat_in_wrist.expand_as(wrist_quat)))
        return position, quat

    def fingertips(self):
        """``(N, 2, 5, 3)`` fingertip positions in the env frame."""
        distal = torch.stack([limb.data.body_link_pose_w.torch[:, bodies]
                              for limb, bodies in zip(self.limbs, self.distal_bodies)], dim=1)
        quat = normalize_canonical_quaternion(distal[..., 3:7])
        origins = self.scene.env_origins[:, None, None, :]
        return distal[..., 0:3] - origins + quat_rotate(quat, self.tip_offsets.expand_as(distal[..., 0:3]))

    def reference_actions(self):
        """The feed-forward actions carrying the current targets onto the next Reference frame.

        Not used by training: a check that the action interface and the drives can
        follow the Reference at all (scripts/evaluate.py --replay).
        """
        following = (self.reference_index + 1).clamp(max=self.reference.last_index)
        actions = torch.zeros((self.num_envs, NUM_LIMBS, ACTIONS_PER_LIMB), device=self.device)
        for limb, side in enumerate(limbs.SIDES):
            pose, _ = self.kinematics[limb].palm_pose_and_jacobian(self.position_targets[:, limb, :NA])
            rotation = self.base_rotation[limb]
            position = pose[:, :3, 3] @ rotation.T + torch.tensor(limbs.base_pose(side)[0], device=self.device)
            quat = matrix_to_quat(rotation @ pose[:, :3, :3])
            linear = self.ref("palm_position", following)[:, limb] - position
            error = normalize_canonical_quaternion(quat_multiply(
                self.ref("palm_quat", following)[:, limb], quat * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=self.device)))
            norm = error[:, :3].norm(dim=-1, keepdim=True)
            angular = error[:, :3] / norm.clamp_min(1e-9) * 2.0 * torch.atan2(norm, error[:, 3:4])
            actions[:, limb, 0:3] = linear / (self.cfg.arm_translation_speed_m_per_s * self.step_dt)
            actions[:, limb, 3:6] = angular / (self.cfg.arm_rotation_speed_rad_per_s * self.step_dt)
            actions[:, limb, 6:] = ((self.ref("q", following)[:, limb, NA:] - self.default_hand[limb])
                                    / self.cfg.hand_action_scale)
        return actions.view(self.num_envs, -1)

    def normalize_positions(self, q):
        return (2.0 * (q - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0).clamp(-1.0, 1.0)

    def hand_actions_for(self, hand_q):
        return (hand_q - self.default_hand) / self.cfg.hand_action_scale

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _arm_targets(self, twist_actions, arm_q, previous):
        """IK step per Limb: world-frame twist actions ``(N, 2, 6)`` -> arm targets ``(N, 2, 6)``."""
        max_translation = self.cfg.arm_translation_speed_m_per_s * self.step_dt
        max_rotation = self.cfg.arm_rotation_speed_rad_per_s * self.step_dt
        linear = saturate_direction_preserving(twist_actions[..., 0:3] * max_translation, max_translation)
        angular = saturate_direction_preserving(twist_actions[..., 3:6] * max_rotation, max_rotation)
        # The Jacobian is in each Limb's base_link frame: v_base = R^T v_world.
        rotation_t = self.base_rotation.transpose(-1, -2)
        linear = torch.einsum("lij,nlj->nli", rotation_t, linear)
        angular = torch.einsum("lij,nlj->nli", rotation_t, angular)
        twist = torch.cat((linear, angular), dim=-1)
        targets = []
        for limb in range(NUM_LIMBS):
            _, jacobian = self.kinematics[limb].palm_pose_and_jacobian(arm_q[:, limb])
            delta = damped_least_squares_step(jacobian, twist[:, limb], self.cfg.ik_damping)
            delta = delta.clamp(-self.cfg.ik_max_joint_delta_rad, self.cfg.ik_max_joint_delta_rad)
            target = torch.maximum(torch.minimum(previous[:, limb] + delta, self.joint_upper[limb, :NA]),
                                   self.joint_lower[limb, :NA])
            applied = target - previous[:, limb]
            self.applied_arm_delta[:, limb] = applied
            achieved = torch.bmm(jacobian, applied.unsqueeze(-1)).squeeze(-1)
            self.ik_residual[:, limb] = (twist[:, limb] - achieved).norm(dim=-1)
            targets.append(target)
        return torch.stack(targets, dim=1)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions)
        # First-order low-pass on the action the environment executes (simtoolreal's
        # action_filter_alpha; 1 = raw). Rewards keep scoring the raw actions.
        self.filtered_actions.lerp_(self.actions, self.cfg.action_filter_alpha)
        per_limb = self.filtered_actions.view(self.num_envs, NUM_LIMBS, ACTIONS_PER_LIMB)
        arm = self._arm_targets(self._world_twist(per_limb[..., :6]), self.q[..., :NA], self.position_targets[..., :NA])
        residual = (per_limb[..., 6:] * self.cfg.hand_action_scale).clamp(
            -self.cfg.hand_action_clip_rad, self.cfg.hand_action_clip_rad)
        hand = torch.maximum(torch.minimum(self.default_hand + residual, self.joint_upper[:, NA:]),
                             self.joint_lower[:, NA:])
        self.previous_targets_2.copy_(self.previous_targets)
        self.previous_targets.copy_(self.position_targets)
        self.position_targets.copy_(torch.cat((arm, hand), dim=-1))

    def _apply_action(self):
        delta = (self.position_targets - self.applied_targets).clamp(-self.target_slew_per_step,
                                                                     self.target_slew_per_step)
        self.applied_targets.add_(delta)
        for limb, articulation in enumerate(self.limbs):
            articulation.actuators.target_command.set_position_index(value=self.applied_targets[:, limb].contiguous())

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device).long()
        if env_ids.numel() == 0:
            return
        super()._reset_idx(env_ids.to(torch.int32))
        count = env_ids.numel()
        if self.cfg.rsi_start_index is None:
            start = torch.randint(0, self.reference.last_index, (count,), device=self.device)
            at_beginning = torch.rand(count, device=self.device) < self.cfg.rsi_start_probability
            start = torch.where(at_beginning, torch.zeros_like(start), start)
        else:
            start = torch.full((count,), int(self.cfg.rsi_start_index), dtype=torch.long, device=self.device)
        self._before_reset(env_ids)
        q, dq = self.ref("q", start, env_ids), self.ref("dq", start, env_ids)       # (count, 2, 26)
        self.reference_index[env_ids] = start
        self.palm_violation_steps[env_ids] = 0
        self.hand_violation_steps[env_ids] = 0
        self.extra_violation_steps[env_ids] = 0
        self.suppress_arm_action_rate[env_ids] = True
        reset_action = torch.zeros((count, NUM_LIMBS, ACTIONS_PER_LIMB), device=self.device)
        reset_action[..., 6:] = self.hand_actions_for(q[..., NA:])
        self.actions[env_ids] = reset_action.view(count, -1)
        self.previous_actions[env_ids] = reset_action.view(count, -1)
        self.filtered_actions[env_ids] = reset_action.view(count, -1)
        self.position_targets[env_ids] = q
        self.previous_targets[env_ids] = q
        self.previous_targets_2[env_ids] = q
        self.applied_targets[env_ids] = q
        self.applied_arm_delta[env_ids] = 0.0
        self.ik_residual[env_ids] = 0.0
        ids = env_ids.to(torch.int32)
        for limb, articulation in enumerate(self.limbs):
            articulation.write_joint_state_to_sim_index(position=q[:, limb].contiguous(),
                                                        velocity=dq[:, limb].contiguous(), env_ids=ids)
            articulation.actuators.target_command.set_position_index(value=q[:, limb].contiguous(), env_ids=ids)
        for values in self.episode_sums.values():
            values[env_ids] = 0.0
        self._reset_extras(env_ids, start)

    # ------------------------------------------------------------------
    # Observations, rewards, terminations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        q, dq = self.q, self.dq
        palm_position, palm_quat = self.palm_pose()
        tips_palm = quat_rotate_inverse(palm_quat.unsqueeze(2).expand(-1, -1, 5, -1),
                                        self.fingertips() - palm_position.unsqueeze(2))
        palm_position, palm_quat = self._observed_palm_pose(palm_position, palm_quat)
        q, targets, dq = self._observed_joints(q, self.position_targets, dq)
        per_limb = torch.cat((
            self.normalize_positions(q),
            targets,
            dq,
            palm_position,
            quat_to_rotation_6d(palm_quat),
            tips_palm.flatten(2),
        ), dim=-1)
        phase = (self.reference_index.float() / self.reference.last_index).unsqueeze(1)
        observation = torch.cat((per_limb.flatten(1), phase), dim=1)
        extra = self._extra_observations()
        if extra is not None:
            observation = torch.cat((observation, extra), dim=1)
        return {"policy": torch.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)}

    def _compute_metrics(self):
        cfg = self.cfg
        index = self.reference_index
        q, dq = self.q, self.dq
        reference_q, reference_dq = self.ref("q", index), self.ref("dq", index)
        palm_position, palm_quat = self.palm_pose()
        palm_mse = (palm_keypoints(palm_position, palm_quat, PALM_LEVER_ARM_M)
                    - self.ref("palm_keypoints", index)).square().sum(-1).mean(-1)          # (N, 2)
        tip_error = (self.fingertips() - self.ref("fingertips", index)).norm(dim=-1).mean(-1)
        hand_q_error = q[..., NA:] - reference_q[..., NA:]
        hand_dq_error = dq[..., NA:] - reference_dq[..., NA:]
        action_delta = (self.actions - self.previous_actions).view(self.num_envs, NUM_LIMBS, ACTIONS_PER_LIMB)
        hand_action_rate = action_delta[..., 6:] - reference_dq[..., NA:] * self.step_dt / cfg.hand_action_scale
        arm_action_rate = torch.where(self.suppress_arm_action_rate[:, None, None],
                                      torch.zeros_like(action_delta[..., :6]), action_delta[..., :6])
        terms = {
            "palm_keypoint": (cfg.palm_keypoint_weight, gaussian(palm_mse, cfg.palm_keypoint_std_m)),
            "hand_position": (cfg.hand_position_weight,
                              gaussian(hand_q_error.square().mean(-1), cfg.hand_position_std_rad)),
            "hand_velocity": (cfg.hand_velocity_weight,
                              gaussian(hand_dq_error.square().mean(-1), cfg.hand_velocity_std_rad_per_s)),
            "hand_action_rate": (cfg.hand_action_rate_weight,
                                 gaussian(hand_action_rate.square().mean(-1), cfg.hand_action_rate_std)),
            "arm_action_rate": (cfg.arm_action_rate_weight,
                                gaussian(arm_action_rate.square().mean(-1), cfg.arm_action_rate_std)),
            "arm_joint_rate": (cfg.arm_joint_rate_weight,
                               gaussian(self.applied_arm_delta.square().mean(-1), cfg.arm_joint_rate_std_rad)),
        }
        # Acceleration of the commanded targets [rad/s^2]: what makes the drives shake.
        target_acceleration = (self.position_targets - 2.0 * self.previous_targets + self.previous_targets_2) \
            / self.step_dt ** 2
        arm_target_acc = target_acceleration[..., :NA].square().mean(-1)
        hand_target_acc = target_acceleration[..., NA:].square().mean(-1)
        if cfg.target_acceleration_weight > 0.0:
            # Linear and saturated, not Gaussian: a chattering policy sits at hundreds of
            # rad/s^2, where a Gaussian is flat and gives no direction to improve.
            limit = cfg.target_acceleration_clip
            penalty = 0.5 * ((arm_target_acc.sqrt() / cfg.arm_target_acceleration_scale).clamp(max=limit)
                             + (hand_target_acc.sqrt() / cfg.hand_target_acceleration_scale).clamp(max=limit))
            terms["target_acceleration"] = (-cfg.target_acceleration_weight, penalty)
        reward = sum(weight * value.mean(dim=1) for weight, value in terms.values())
        metrics = {f"reward_{name}": value.mean(dim=1) for name, (_, value) in terms.items()}
        palm_error = palm_mse.sqrt()
        hand_error = hand_q_error.abs().amax(dim=-1)
        metrics.update({
            "palm_keypoint_error_m": palm_error.amax(dim=1),
            "palm_position_error_m": (palm_position - self.ref("palm_position", index)).norm(dim=-1).amax(dim=1),
            "fingertip_error_m": tip_error.amax(dim=1),
            "hand_max_error_rad": hand_error.amax(dim=1),
            "hand_rms_error_rad": hand_q_error.square().mean(-1).sqrt().amax(dim=1),
            "ik_residual": self.ik_residual.amax(dim=1),
            "arm_target_acc": arm_target_acc.sqrt().amax(dim=1),
            "hand_target_acc": hand_target_acc.sqrt().amax(dim=1),
        })
        return reward, metrics, palm_error, hand_error

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.reference_index.add_(1).clamp_(max=self.reference.last_index)
        reward, metrics, palm_error, hand_error = self._compute_metrics()
        self.suppress_arm_action_rate.fill_(False)
        palm_violation = (palm_error > self.cfg.palm_keypoint_threshold_m).any(dim=1)
        hand_violation = (hand_error > self.cfg.hand_position_threshold_rad).any(dim=1)
        self.palm_violation_steps = torch.where(palm_violation, self.palm_violation_steps + 1, 0)
        self.hand_violation_steps = torch.where(hand_violation, self.hand_violation_steps + 1, 0)
        early = (self.palm_violation_steps >= self.cfg.grace_steps) | (self.hand_violation_steps >= self.cfg.grace_steps)
        # A blown-up world ends its episode; nothing non-finite reaches the learner.
        finite = torch.isfinite(self.q).flatten(1).all(dim=1) & torch.isfinite(self.dq).flatten(1).all(dim=1)
        early = early | ~finite
        extra = self._extra_violation()
        if extra is not None:
            self.extra_violation_steps = torch.where(extra, self.extra_violation_steps + 1, 0)
            early = early | (self.extra_violation_steps >= self.cfg.grace_steps)
        timeout = self.reference_index >= self.reference.last_index
        self._reward = torch.nan_to_num(torch.where(finite, reward, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
        self._metrics = metrics
        self._early = early
        return early, timeout & ~early

    def _get_rewards(self) -> torch.Tensor:
        done = self.reset_buf
        sums = self.episode_sums
        for name, value in self._metrics.items():
            if name not in sums:
                sums[name] = torch.zeros(self.num_envs, device=self.device)
            sums[name] += torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        self.extras = {"metrics": self._metrics, "reference_index": self.reference_index.clone()}
        if bool(done.any()):
            length = self.episode_length_buf[done].clamp(min=1).float()
            log = {f"Episode/{name}": (values[done] / length).mean() for name, values in sums.items()}
            log["Episode/early_termination_fraction"] = self._early[done].float().mean()
            log["Episode/length"] = length.mean()
            log["Episode/final_phase"] = (self.reference_index[done].float() / self.reference.last_index).mean()
            self.extras["log"] = log
        return self._reward
