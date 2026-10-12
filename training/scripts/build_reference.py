#!/usr/bin/env python3
"""Builds a Reference (CONTEXT.md) from a Demonstration.

One Reference frame per Demonstration sample (recorded at 50 Hz, the control
rate). Per Limb it stores the measured joints, their velocities, and the Hand-base
pose and fingertips in the world, computed with the Limb URDF's forward
kinematics placed at the Scene's Base pose. The Hand-base pose MuJoCo recorded is
used as a check of that placement.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/build_reference.py \
        teleoperation/demonstrations/20261012_093615_lift_cube_trimmed.npz [--name lift_cube]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bimanual_training import REFERENCES_DIR, limbs
from bimanual_training.kinematics import LimbKinematics
from bimanual_training.rotations import matrix_to_quat, quat_conjugate, quat_multiply

CONTROL_DT = 0.02
PLACEMENT_TOLERANCE_M = 0.005


def limb_reference(side, data):
    q = np.concatenate([data[f"{side}_arm_q_measured"], data[f"{side}_hand_q_measured"]], axis=1)
    kinematics = LimbKinematics(limbs.urdf_path(side), limbs.ARM_JOINTS, limbs.HAND_JOINTS, dtype=torch.float64)
    # MuJoCo's joint limits are soft: a pressed joint sits a few mrad past its range.
    lower, upper = kinematics.lower_limits.numpy(), kinematics.upper_limits.numpy()
    clipped = np.clip(q, lower, upper)
    if np.abs(clipped - q).max() > 0:
        j = int(np.abs(clipped - q).max(axis=0).argmax())
        print(f"{side}: clipped to the joint limits, at most {np.abs(clipped - q).max() * 1000:.1f} mrad "
              f"({limbs.JOINTS[j]})")
    q = clipped
    palm, tips = kinematics.palm_and_fingertips(torch.as_tensor(q, dtype=torch.float64))
    position, _ = limbs.base_pose(side)
    rotation = limbs.base_rotation(side, dtype=torch.float64)
    origin = torch.tensor(position, dtype=torch.float64)
    palm_position = palm[:, :3, 3] @ rotation.T + origin
    palm_quat = matrix_to_quat(rotation @ palm[:, :3, :3])                # xyzw
    tips_world = tips @ rotation.T + origin
    return {
        "q": q.astype(np.float32),
        "dq": np.gradient(q, CONTROL_DT, axis=0).astype(np.float32),
        "palm_position": palm_position.numpy().astype(np.float32),
        "palm_quat": palm_quat.numpy().astype(np.float32),
        "fingertips": tips_world.numpy().astype(np.float32),
    }


def placement_error(reference, recorded):
    """Largest position [m] and angle [rad] gap to the Hand-base pose MuJoCo recorded."""
    position = np.linalg.norm(reference["palm_position"] - recorded[:, :3], axis=1).max()
    relative = quat_multiply(quat_conjugate(torch.as_tensor(recorded[:, 3:7], dtype=torch.float64)),
                             torch.as_tensor(reference["palm_quat"], dtype=torch.float64))
    angle = (2.0 * torch.atan2(relative[:, :3].norm(dim=1), relative[:, 3].abs())).max().item()
    return float(position), angle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("demonstration", type=Path)
    parser.add_argument("--name", default="lift_cube")
    args = parser.parse_args()

    data = np.load(args.demonstration)
    steps = np.diff(data["time"])
    if abs(steps.mean() - CONTROL_DT) > 1e-3:
        raise SystemExit(f"Demonstration sampled every {steps.mean():.4f} s, expected {CONTROL_DT}")

    out = {"dt": np.float32(CONTROL_DT), "object_pose": data["object_pose"].astype(np.float32)}
    for side in limbs.SIDES:
        reference = limb_reference(side, data)
        position_error, angle_error = placement_error(reference, data[f"{side}_hand_base_measured"])
        print(f"{side}: Hand base vs recorded: {1000 * position_error:.2f} mm, {np.degrees(angle_error):.2f} deg")
        if position_error > PLACEMENT_TOLERANCE_M:
            raise SystemExit(f"{side}: forward kinematics does not match the recorded Hand base")
        out.update({f"{side}_{k}": v for k, v in reference.items()})

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    path = REFERENCES_DIR / f"{args.name}.npz"
    np.savez_compressed(path, **out)
    path.with_suffix(".json").write_text(json.dumps({
        "demonstration": args.demonstration.name,
        "frames": int(len(data["time"])),
        "duration_s": float(data["time"][-1] - data["time"][0]),
        "dt": CONTROL_DT,
    }, indent=2))
    print(f"{len(data['time'])} frames -> {path}")


if __name__ == "__main__":
    main()
