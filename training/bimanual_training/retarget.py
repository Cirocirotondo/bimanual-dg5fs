"""Retargeting a Reference to Object placements: both Arms re-solved by IK.

For every placement the Hand-base poses of the Reference move rigidly with the
Object (placement.py); each Arm is then re-solved frame by frame with damped
least squares on its URDF chain (kinematics.LimbKinematics), batched over all
placements at once. The first frame starts from the Demonstration's joints and
every later frame from the previous solution, so the Arm stays in the
Demonstration's joint family (CONTEXT.md, Transform bank). Hand joints are not
touched: the move is rigid, so the fingers keep the Demonstration's.

From simtoolreal_newton's retarget.py (solve_palm_ik, retarget_clip), widened to
two Limbs and to a batch of placements.
"""
import torch

from bimanual_training import limbs
from bimanual_training.kinematics import LimbKinematics
from bimanual_training.placement import transform_points, transform_quaternions
from bimanual_training.rotations import matrix_to_quat, quat_conjugate, quat_multiply, quat_to_matrix

IK_DAMPING = 0.02
FIRST_FRAME_ITERATIONS = 200
FRAME_ITERATIONS = 40
# The first frame is reached by moving the placement there in this many steps from
# the Demonstration's own (a continuation): one jump of up to 45 degrees from the
# Demonstration's joints can leave damped least squares short of a pose the
# Arm reaches easily, without leaving the Demonstration's joint family.
CONTINUATION_STEPS = 12
TOLERANCE = 1e-4                        # m and rad: a frame is solved below both


def rotation_error(target, current):
    """``(..., 3)`` rotation vector taking ``current`` to ``target`` (xyzw), in their common frame."""
    error = quat_multiply(target, quat_conjugate(current))
    error = torch.where(error[..., 3:4] < 0.0, -error, error)
    vector = error[..., :3]
    norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, error[..., 3:4])
    return vector / norm.clamp_min(1e-12) * angle


class PlacementRetargeter:
    """Retargets one Reference's Arms to batches of placements."""

    def __init__(self, reference_npz, device="cpu", dtype=torch.float64):
        import numpy as np

        data = np.load(reference_npz)
        self.device, self.dtype = torch.device(device), dtype

        def tensor(key):
            return torch.as_tensor(data[key], dtype=dtype, device=self.device)

        self.object_pose = tensor("object_pose")                                    # (F, 7)
        self.pivot = self.object_pose[0, :3].clone()
        self.frames = self.object_pose.shape[0]
        self.limbs = []
        for side in limbs.SIDES:
            kinematics = LimbKinematics(limbs.urdf_path(side), limbs.ARM_JOINTS, limbs.HAND_JOINTS,
                                        device=self.device, dtype=dtype)
            position, _ = limbs.base_pose(side)
            self.limbs.append({
                "side": side,
                "kinematics": kinematics,
                "q": tensor(f"{side}_q"),                                           # (F, 26)
                "palm_position": tensor(f"{side}_palm_position"),                   # (F, 3) world
                "palm_quat": tensor(f"{side}_palm_quat"),                           # (F, 4) xyzw
                "base_position": torch.tensor(position, dtype=dtype, device=self.device),
                "base_rotation": limbs.base_rotation(side, device=self.device, dtype=dtype),
                "lower": kinematics.lower_limits[: limbs.NUM_ARM].to(self.device, dtype),
                "upper": kinematics.upper_limits[: limbs.NUM_ARM].to(self.device, dtype),
            })

    def palm_targets(self, limb, translation, yaw, frames):
        """Target Hand-base poses in the Limb's base_link frame: positions (P, k, 3), xyzw (P, k, 4)."""
        position = limb["palm_position"][frames].unsqueeze(0)                     # (1, k, 3)
        quat = limb["palm_quat"][frames].unsqueeze(0)
        world_position = transform_points(position, translation[:, None, :], yaw[:, None], self.pivot)
        world_quat = transform_quaternions(quat.expand(len(yaw), -1, -1), yaw[:, None])
        rotation_t = limb["base_rotation"].T
        base_position = (world_position - limb["base_position"]) @ rotation_t.T
        base_quat = quat_multiply(matrix_to_quat(rotation_t).expand_as(world_quat), world_quat)
        return base_position, base_quat

    def solve(self, translation, yaw, frame_stride=1, progress=None):
        """Retarget both Arms to placements ``translation (P, 3)``, ``yaw (P,)``.

        Returns per Limb: arm q (P, k, 6) and the residuals (P, k) in m and rad, for
        the frames ``range(0, F, frame_stride)``.
        """
        translation = translation.to(self.device, self.dtype)
        yaw = yaw.to(self.device, self.dtype)
        frames = torch.arange(0, self.frames, frame_stride, device=self.device)
        results = []
        for limb in self.limbs:
            kinematics = limb["kinematics"]
            target_position, target_quat = self.palm_targets(limb, translation, yaw, frames)
            count = len(yaw)
            q = limb["q"][0, : limbs.NUM_ARM].expand(count, -1).clone()
            arm_q = torch.empty((count, len(frames), limbs.NUM_ARM), dtype=self.dtype, device=self.device)
            position_residual = torch.empty((count, len(frames)), dtype=self.dtype, device=self.device)
            rotation_residual = torch.empty_like(position_residual)
            # Smallest singular value of the Hand-base Jacobian: how close the Arm is to a
            # singularity (the policy's IK step loses lambda^2 / (sigma^2 + lambda^2) of a command).
            sigma_min = torch.empty_like(position_residual)
            eye = torch.eye(6, dtype=self.dtype, device=self.device)
            def iterate(q, goal_position, goal_quat, iterations):
                for _ in range(iterations):
                    pose, jacobian = kinematics.palm_pose_and_jacobian(q)
                    current_quat = matrix_to_quat(pose[:, :3, :3])
                    linear = goal_position - pose[:, :3, 3]
                    angular = rotation_error(goal_quat, current_quat)
                    active = (linear.norm(dim=-1) > TOLERANCE) | (angular.norm(dim=-1) > TOLERANCE)
                    if not bool(active.any()):
                        break
                    error = torch.cat((linear, angular), dim=-1)
                    jt = jacobian.transpose(1, 2)
                    step = (jt @ torch.linalg.solve(jacobian @ jt + IK_DAMPING ** 2 * eye,
                                                    error.unsqueeze(-1))).squeeze(-1)
                    q = torch.where(active.unsqueeze(-1), q + step, q)
                    q = torch.maximum(torch.minimum(q, limb["upper"]), limb["lower"])
                return q

            for fraction in torch.linspace(1.0 / CONTINUATION_STEPS, 1.0, CONTINUATION_STEPS).tolist()[:-1]:
                partial_position, partial_quat = self.palm_targets(limb, translation * fraction, yaw * fraction,
                                                                   frames[:1])
                q = iterate(q, partial_position[:, 0], partial_quat[:, 0], FRAME_ITERATIONS)
            for k in range(len(frames)):
                iterations = FIRST_FRAME_ITERATIONS if k == 0 else FRAME_ITERATIONS
                q = iterate(q, target_position[:, k], target_quat[:, k], iterations)
                pose, jacobian = kinematics.palm_pose_and_jacobian(q)
                arm_q[:, k] = q
                sigma_min[:, k] = torch.linalg.svdvals(jacobian)[:, -1]
                position_residual[:, k] = (target_position[:, k] - pose[:, :3, 3]).norm(dim=-1)
                rotation_residual[:, k] = rotation_error(target_quat[:, k],
                                                         matrix_to_quat(pose[:, :3, :3])).norm(dim=-1)
                if progress is not None:
                    progress(limb["side"], k, len(frames))
            results.append({"arm_q": arm_q, "position_residual": position_residual,
                            "rotation_residual": rotation_residual, "sigma_min": sigma_min,
                            "lower": limb["lower"], "upper": limb["upper"]})
        return frames, results

    def hand_q(self, frames):
        """Hand joints of both Limbs (k, 2, 20): unchanged by a placement."""
        return torch.stack([limb["q"][frames, limbs.NUM_ARM:] for limb in self.limbs], dim=1)

    def object_track(self, translation, yaw, frames):
        """Object poses (P, k, 7) xyz + xyzw for the placements."""
        position = transform_points(self.object_pose[frames, :3].unsqueeze(0), translation[:, None, :],
                                    yaw[:, None], self.pivot)
        quat = transform_quaternions(self.object_pose[frames, 3:7].unsqueeze(0).expand(len(yaw), -1, -1),
                                     yaw[:, None])
        return torch.cat((position, quat), dim=-1)
