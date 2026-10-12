#!/usr/bin/env python3
"""Headless check that both Limb URDFs load and step in Newton (no Isaac Lab).

Places each Limb at its Base pose, drives every joint to the Home pose with the
same position gains as the MuJoCo Scene, steps the MuJoCo-Warp solver, and fails
if the state goes non-finite or drifts away from Home.

    uv run scripts/newton_smoke.py [--steps 500] [--device cuda:0]
"""
import argparse
import sys

import numpy as np
import warp as wp
import newton
from scipy.spatial.transform import Rotation as R

from bimanual_dg5fs import scene

MAX_DRIFT = 0.05  # rad; no gravity compensation here, so the Arms sag a little


def joint_settings():
    """{joint label: (home q, kp, kv, armature)} for every actuated joint."""
    settings = {}
    for side in scene.SIDES:
        prefix = f"ur5e_{side}_dg5fs/"
        for i, name in enumerate(scene.ARM_JOINTS):
            settings[prefix + name] = (scene.HOME_ARM_Q[side][i], scene.ARM_KP[i],
                                       scene.ARM_KV[i], scene.ARM_ARMATURE)
        for name in scene.HAND_JOINTS:
            settings[prefix + name] = (0.0, scene.HAND_KP, scene.HAND_KV, scene.HAND_ARMATURE)
    return settings


def build_model(device):
    builder = newton.ModelBuilder()
    for side in scene.SIDES:
        yaw = scene.BASE_YAW_DEG[side] - scene.URDF_BASE_LINK_TO_BASE_YAW_DEG
        q = R.from_euler("z", yaw, degrees=True).as_quat()  # x, y, z, w (Warp order)
        builder.add_urdf(
            str(scene.limb_urdf(side)),
            xform=wp.transform(wp.vec3(*scene.BASE_POS[side]), wp.quat(*q)),
            floating=False,
            enable_self_collisions=False,
        )
    builder.add_ground_plane(height=scene.TABLE_TOP_Z)

    settings = joint_settings()
    seen = set()
    for j, label in enumerate(builder.joint_label):
        if label not in settings:
            continue
        home, kp, kv, armature = settings[label]
        q, dof = builder.joint_q_start[j], builder.joint_qd_start[j]
        builder.joint_q[q] = home
        builder.joint_target_q[q] = home
        builder.joint_target_ke[dof] = kp
        builder.joint_target_kd[dof] = kv
        builder.joint_armature[dof] = armature
        seen.add(label)
    missing = sorted(set(settings) - seen)
    if missing:
        sys.exit(f"FAIL: joints not found in Newton model: {missing}")
    return builder.finalize(device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model = build_model(args.device)
    revolute = int((model.joint_type.numpy() == int(newton.JointType.REVOLUTE)).sum())
    print(f"Newton {newton.__version__} on {model.device}: {model.articulation_count} "
          f"articulations, {model.body_count} bodies, {revolute} revolute joints, "
          f"{model.joint_dof_count} dofs, {model.shape_count} shapes")
    if model.articulation_count != 2 or revolute != 52:
        sys.exit("FAIL: expected 2 Limbs with 52 revolute joints")

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    q_home = state_0.joint_q.numpy().copy()

    solver = newton.solvers.SolverMuJoCo(model)
    dt = 0.002
    for _ in range(args.steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    q = state_0.joint_q.numpy()
    if not np.all(np.isfinite(q)):
        sys.exit("FAIL: non-finite joint state")
    drift = float(np.abs(q - q_home).max())
    print(f"stepped {args.steps} x {dt} s; max |q - q_home| = {drift:.4f} rad")
    if drift > MAX_DRIFT:
        sys.exit(f"FAIL: drift above {MAX_DRIFT} rad")
    print("OK")


if __name__ == "__main__":
    main()
