#!/usr/bin/env python3
"""Builds a Reference (CONTEXT.md) from a Demonstration.

One Reference frame per Demonstration sample (recorded at 50 Hz, the control
rate); with --resample, one per 20 ms of the recorded time stamps, for a
Demonstration whose control loop did not hold 50 Hz (joints interpolated linearly,
poses with slerp). Per Limb it stores the measured joints, their velocities, and the Hand-base
pose and fingertips in the world, computed with the Limb URDF's forward
kinematics placed at the Scene's Base pose. The Hand-base pose MuJoCo recorded is
used as a check of that placement. The Object's pose is stored as recorded; an
articulated Object's joints (the book's hinge) are low-passed like the Limbs' and
clipped to their range.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/build_reference.py \
        teleoperation/demonstrations/20261012_093615_lift_cube_trimmed.npz [--name lift_cube] [--lowpass-hz 3] \
        [--resample]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bimanual_dg5fs import scene
from bimanual_training import REFERENCES_DIR, limbs
from bimanual_training.kinematics import LimbKinematics
from bimanual_training.rotations import matrix_to_quat, quat_conjugate, quat_multiply

CONTROL_DT = 0.02
PLACEMENT_TOLERANCE_M = 0.005


def resample(data):
    """The Demonstration's samples on a uniform CONTROL_DT grid of its time stamps: the
    joint arrays interpolated linearly, poses (x y z qx qy qz qw) with slerp; other
    per-sample arrays (flags, raw inputs) are not used by the Reference and dropped."""
    from scipy.spatial.transform import Rotation, Slerp

    t = np.asarray(data["time"], dtype=np.float64)
    grid = t[0] + CONTROL_DT * np.arange(int(np.floor((t[-1] - t[0]) / CONTROL_DT + 1e-9)) + 1)
    out = {"time": grid}
    for key in data.files:
        value = np.asarray(data[key])
        if key == "time" or value.ndim == 0 or len(value) != len(t) or value.dtype == bool:
            continue
        if key.endswith("_pose") or key.endswith("hand_base_measured") or key.endswith("hand_base_target"):
            if value.shape[1:] != (7,):
                continue
            position = np.stack([np.interp(grid, t, value[:, i]) for i in range(3)], axis=1)
            quat = Slerp(t, Rotation.from_quat(value[:, 3:7]))(grid).as_quat()
            out[key] = np.concatenate([position, quat], axis=1)
        elif key.endswith("_q_measured") or key.endswith("_q_commanded") or key == "object_joint_q":
            flat = value.reshape(len(t), -1)
            out[key] = np.stack([np.interp(grid, t, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out


class _Arrays(dict):
    """A dict that answers ``.files`` like the npz it stands in for."""
    @property
    def files(self):
        return list(self.keys())


def lowpass(q, hz):
    """Zero-phase 2nd-order Butterworth low-pass along time (no lag, so the
    Reference stays aligned with the Object it was recorded with)."""
    from scipy.signal import butter, filtfilt

    b, a = butter(2, hz, fs=1.0 / CONTROL_DT)
    return filtfilt(b, a, q, axis=0)


def limb_reference(side, data, lowpass_hz=None):
    q = np.concatenate([data[f"{side}_arm_q_measured"], data[f"{side}_hand_q_measured"]], axis=1)
    if lowpass_hz:
        # The glove's tremor is in the Demonstration (65% of the hand joints' velocity
        # power above 4 Hz in lift_cube_2); a policy imitating it would shake the hands.
        q = lowpass(q, lowpass_hz)
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


def object_joint_reference(data, object_name, lowpass_hz=None):
    """An articulated Object's joints ``(T, J)`` (the Scene's OBJECT_JOINTS order), low-passed like
    the Limbs' and clipped to the joint ranges; None when the Demonstration records none."""
    if "object_joint_q" not in data.files:
        return None
    q = np.asarray(data["object_joint_q"], dtype=np.float64)
    ranges = scene.object_joint_ranges(object_name)
    if q.shape[1] != len(ranges):
        raise SystemExit(f"the Demonstration records {q.shape[1]} Object joints, the Scene's {object_name} has "
                         f"{len(ranges)} ({list(ranges)})")
    if lowpass_hz:
        q = lowpass(q, lowpass_hz)
    lower, upper = (np.array([r[i] for r in ranges.values()]) for i in (0, 1))
    clipped = np.clip(q, lower, upper)
    for j, name in enumerate(ranges):
        # MuJoCo's limits are soft (a pressed cover sits a little past closed), and the
        # low-pass can overshoot a limit the cover rests against.
        print(f"{name}: {np.degrees(clipped[:, j].min()):.1f} to {np.degrees(clipped[:, j].max()):.1f} deg, "
              f"clipped to the range by at most {np.degrees(np.abs(clipped[:, j] - q[:, j]).max()):.2f} deg")
    return clipped


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
    parser.add_argument("--lowpass-hz", type=float, default=None,
                        help="zero-phase low-pass of the joints before forward kinematics (e.g. 3)")
    parser.add_argument("--resample", action="store_true",
                        help="resample on a uniform 50 Hz grid of the recorded time stamps")
    args = parser.parse_args()

    data = np.load(args.demonstration)
    recorded = len(data["time"])
    steps = np.diff(data["time"])
    print(f"Demonstration: {recorded} samples, step mean {1000 * steps.mean():.2f} ms, "
          f"median {1000 * np.median(steps):.2f} ms, max {1000 * steps.max():.1f} ms")
    if args.resample:
        data = _Arrays(resample(data))
        print(f"resampled to {len(data['time'])} frames every {1000 * CONTROL_DT:.0f} ms")
    elif abs(steps.mean() - CONTROL_DT) > 1e-3:
        raise SystemExit(f"Demonstration sampled every {steps.mean():.4f} s, expected {CONTROL_DT}: "
                         f"use --resample")

    out = {"dt": np.float32(CONTROL_DT), "object_pose": data["object_pose"].astype(np.float32)}
    # The Scene's other free bodies (the cube next to the basin), stored as recorded.
    for key in data.files:
        if key.endswith("_pose") and key != "object_pose":
            out[key] = np.asarray(data[key]).astype(np.float32)
    for side in limbs.SIDES:
        reference = limb_reference(side, data, args.lowpass_hz)
        # The placement check is on the unfiltered joints: it validates the Base placement
        # and the kinematics, not the low-pass (reported separately).
        raw = limb_reference(side, data) if args.lowpass_hz else reference
        position_error, angle_error = placement_error(raw, data[f"{side}_hand_base_measured"])
        print(f"{side}: Hand base vs recorded: {1000 * position_error:.2f} mm, {np.degrees(angle_error):.2f} deg")
        if position_error > PLACEMENT_TOLERANCE_M:
            raise SystemExit(f"{side}: forward kinematics does not match the recorded Hand base")
        if args.lowpass_hz:
            shift, turn = placement_error(reference, data[f"{side}_hand_base_measured"])
            print(f"{side}: the low-pass moves the Hand base by at most {1000 * shift:.2f} mm, {np.degrees(turn):.2f} deg")
        out.update({f"{side}_{k}": v for k, v in reference.items()})

    sidecar = args.demonstration.with_suffix(".json")
    scene_info = json.loads(sidecar.read_text()).get("scene", {}) if sidecar.is_file() else {}
    object_name = scene_info.get("object", "cube")
    object_joint_q = object_joint_reference(data, object_name, args.lowpass_hz)
    if object_joint_q is not None:
        out["object_joint_q"] = object_joint_q.astype(np.float32)

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    path = REFERENCES_DIR / f"{args.name}.npz"
    np.savez_compressed(path, **out)
    path.with_suffix(".json").write_text(json.dumps({
        "demonstration": args.demonstration.name,
        "object": object_name,
        "object_joints": list(scene.OBJECT_JOINTS[object_name]) if object_joint_q is not None else [],
        "frames": int(len(data["time"])),
        "duration_s": float(data["time"][-1] - data["time"][0]),
        "dt": CONTROL_DT,
        "lowpass_hz": args.lowpass_hz,
        "resampled_from_samples": recorded if args.resample else None,
        "extra_objects": sorted(k[:-len("_pose")] for k in out if k.endswith("_pose") and k != "object_pose"),
    }, indent=2))
    print(f"{len(data['time'])} frames -> {path}")


if __name__ == "__main__":
    main()
