#!/usr/bin/env python3
"""Runs the Reference once from a start frame and scores the tracking.

    # a trained policy (deterministic: the mean action)
    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/evaluate.py \
        --checkpoint training/logs/motion_imitation/<run>/model_3000.pt [--start 0] \
        [--video rollout.mp4] [--json result.json]
    # no policy: feed-forward actions from the Reference (checks the task is feasible)
    ... training/scripts/evaluate.py --replay

Success (Motion imitation stage): the whole Reference without early termination,
mean Hand-base position error under 2 cm and mean hand joint RMS error under
0.1 rad (worst Limb at each step). With an articulated Object (the book) its
joint error against the Reference (the hinge) is reported too, with the final
angle reached next to the Reference's; it does not enter the success test. For the
bag (10 joints) the error is the worst joint's and the angles the largest of any joint.
With basin_cube (the cube beside the basin) the cube's error against its track is
reported too, with where it ends (height, in the basin or not) and where it is
when the basin is highest; its error enters the success test with the Object's gates.

--video renders the rollout next to the Reference with the MuJoCo Scene
(bimanual_training/video.py). training/scripts/evaluate_run.py does all of this
for a run's checkpoint at several start frames.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")       # before anything imports mujoco

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bimanual_training.launch import add_env_arguments, make_env, parse_settings  # noqa: E402

PALM_SUCCESS_M = 0.02
HAND_SUCCESS_RAD = 0.1
HIGH_FREQUENCY_HZ = 4.0       # joint motion above this counts as vibration
# Training stage 2 (with the Object): its mean and largest distance from the Reference.
OBJECT_MEAN_SUCCESS_M = 0.03
OBJECT_MAX_SUCCESS_M = 0.07   # simtoolreal_newton's gate
# basin_cube: the basin counts as lifted (with the cube in it or not) this far above its start.
LIFTED_M = 0.05


def smoothness(q, dt):
    """Vibration measures of a joint trajectory ``(T, 2, 26)`` sampled every ``dt``.

    ``*_acc_rms``: RMS joint acceleration [rad/s^2], arm and hand apart.
    ``*_hf``: share of the joint-velocity power above HIGH_FREQUENCY_HZ (0..1).
    """
    from bimanual_training import limbs

    out = {}
    if len(q) < 8:
        return out
    velocity = np.diff(q, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    frequencies = np.fft.rfftfreq(len(velocity), dt)
    power = np.abs(np.fft.rfft(velocity - velocity.mean(axis=0), axis=0)) ** 2
    for name, joints in (("arm", slice(0, limbs.NUM_ARM)), ("hand", slice(limbs.NUM_ARM, None))):
        out[f"{name}_acc_rms"] = float(np.sqrt(np.mean(acceleration[..., joints] ** 2)))
        total = power[..., joints].sum()
        out[f"{name}_hf"] = float(power[frequencies > HIGH_FREQUENCY_HZ][..., joints].sum() / max(total, 1e-12))
    return out


def load_policy(env, checkpoint, device):
    import importlib.metadata as metadata

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from bimanual_training.agents import MotionImitationPPORunnerCfg

    agent_cfg = handle_deprecated_rsl_rl_cfg(MotionImitationPPORunnerCfg(), metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    runner = OnPolicyRunner(RslRlVecEnvWrapper(env), agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=device)


def rollout(env, policy, start):
    """One deterministic rollout of a 1-env environment from Reference frame ``start``.

    ``policy`` None runs the feed-forward replay. Returns the scores and, per
    tracked frame, the joints and the Reference frame they were scored against.
    """
    from tensordict import TensorDict

    env.cfg.rsi_start_index = int(start)
    observations, _ = env.reset()
    palm, hand, tips, q, targets, index = [], [], [], [], [], []
    has_object = hasattr(env, "object_pose")
    has_joints = bool(getattr(env, "articulated", False))
    has_cube = bool(getattr(env, "extra_objects", None))       # the cube beside the basin
    cube = {key: [] for key in ("error", "angle", "height", "above", "distance", "basin_height")}
    cube_pose, reference_cube = [], []                          # per traced frame, for the video
    object_error, object_angle, object_height, object_pose = [], [], [], []
    joint_error, joint_q, joint_reference = [], [], []          # per scored step: hinge error, angle, Reference's
    object_joint_q, reference_joint_q = [], []                  # per traced frame, for the video
    reference_q, reference_object = [], []
    terminated_at = None
    # no_grad, not inference_mode: inference tensors left in the environment's
    # buffers would stop a later checkpoint's runner from initialising.
    with torch.no_grad():
        while True:
            if policy is None:
                actions = env.reference_actions()
            else:
                actions = policy(TensorDict(observations, batch_size=[env.num_envs]))
            observations, _, early, timeout, extras = env.step(actions)
            metrics = extras["metrics"]
            palm.append(float(metrics["palm_position_error_m"][0]))
            hand.append(float(metrics["hand_rms_error_rad"][0]))
            tips.append(float(metrics["fingertip_error_m"][0]))
            if has_object:
                object_error.append(float(metrics["object_position_error_m"][0]))
                object_angle.append(float(metrics["object_orientation_error_rad"][0]))
                object_height.append(float(metrics["object_height_m"][0]))
            if has_cube:
                for key, metric in (("error", "cube_position_error_m"), ("angle", "cube_orientation_error_rad"),
                                    ("height", "cube_height_m"), ("above", "cube_above_basin_m"),
                                    ("distance", "cube_basin_distance_m"), ("basin_height", "object_height_m")):
                    cube[key].append(float(metrics[metric][0]))
            if has_joints:
                scored = int(extras["reference_index"][0])
                joint_error.append(float(metrics["object_joint_error_rad"][0]))
                joint_q.append(float(metrics["object_joint_q_rad"][0]))
                joint_reference.append(float(env.reference.object_joint_q[scored, 0]))
            if bool(early[0]):
                terminated_at = int(extras["reference_index"][0])
                break
            if bool(timeout[0]):
                break
            # The state the errors were measured on (an ending step has already reset it).
            q.append(env.q[0].cpu().numpy())
            # The Reference as this episode sees it (moved by the Object placement in stage 3).
            frame = env.reference_index[:1]
            reference_q.append(env.ref("q", frame, env_ids=frame.new_zeros(1))[0].cpu().numpy())
            if has_object:
                reference_object.append(env.ref("object_pose", frame, env_ids=frame.new_zeros(1))[0].cpu().numpy())
            targets.append(env.applied_targets[0].cpu().numpy())
            if has_object:
                position, quat = env.object_pose()
                object_pose.append(torch.cat((position[0], quat[0])).cpu().numpy())
            if has_cube:
                position, quat = env.cube_pose()
                cube_pose.append(torch.cat((position[0], quat[0])).cpu().numpy())
                reference_cube.append(env.ref("cube_pose", frame, env_ids=frame.new_zeros(1))[0].cpu().numpy())
            if has_joints:
                object_joint_q.append(env.object_joint_q()[0].cpu().numpy())
                reference_joint_q.append(env.ref("object_joint_q", frame, env_ids=frame.new_zeros(1))[0].cpu().numpy())
            index.append(int(env.reference_index[0]))
    palm, hand, tips = np.array(palm), np.array(hand), np.array(tips)
    span = env.reference.last_index - int(start)
    result = {
        "start": int(start),
        "frames_tracked": len(palm),
        "frames_to_track": span,
        "terminated_at": terminated_at,
        "palm_mean_m": float(palm.mean()), "palm_max_m": float(palm.max()),
        "hand_rms_mean_rad": float(hand.mean()), "hand_rms_max_rad": float(hand.max()),
        "fingertip_mean_m": float(tips.mean()), "fingertip_max_m": float(tips.max()),
    }
    result["success"] = bool(terminated_at is None and len(palm) >= span
                             and palm.mean() < PALM_SUCCESS_M and hand.mean() < HAND_SUCCESS_RAD)
    if has_object:
        object_error = np.array(object_error)
        result.update({
            "object_mean_m": float(object_error.mean()), "object_max_m": float(object_error.max()),
            "object_angle_mean_rad": float(np.mean(object_angle)),
            "object_lift_m": float(max(object_height) - env.reference.object_pose[0, 2].item()),
            "reference_lift_m": float((env.reference.object_pose[:, 2].max() - env.reference.object_pose[0, 2]).item()),
        })
        result["success"] = bool(result["success"] and object_error.mean() < OBJECT_MEAN_SUCCESS_M
                                 and object_error.max() < OBJECT_MAX_SUCCESS_M)
    if has_joints:
        # The hinge: its error, the angle reached at the last tracked frame and the widest
        # opening, each next to the Reference's over the same frames. The angles come from
        # the trace: an episode's ending step has already reset the hinge to the start frame.
        # With several joints (the bag) the error is the worst joint's and the angles the largest
        # of any joint.
        traced, traced_reference = np.ravel(object_joint_q), np.ravel(reference_joint_q)
        if env.reference.object_joints > 1:
            # Several joints (the bag's bends and twists): the largest angle of any joint at each frame.
            traced = np.abs(np.asarray(object_joint_q)).max(axis=1) if len(object_joint_q) else traced
            traced_reference = (np.abs(np.asarray(reference_joint_q)).max(axis=1) if len(reference_joint_q)
                                else traced_reference)
        if not len(traced):
            traced, traced_reference = np.asarray(joint_q), np.asarray(joint_reference)
        result.update({
            "object_joint_error_mean_rad": float(np.mean(joint_error)),
            "object_joint_error_max_rad": float(np.max(joint_error)),
            "object_joint_final_rad": float(traced[-1]), "reference_joint_final_rad": float(traced_reference[-1]),
            "object_joint_max_rad": float(np.max(traced)), "reference_joint_max_rad": float(np.max(traced_reference)),
        })
    if has_cube:
        result.update(cube_scores(env, cube))
        # The cube is tracked like the Object, with the same gates.
        result["success"] = bool(result["success"] and result["cube_mean_m"] < OBJECT_MEAN_SUCCESS_M
                                 and result["cube_max_m"] < OBJECT_MAX_SUCCESS_M)
    # Vibration of the joints, of the targets sent to the drives, and of the
    # Reference over the same frames: the policy should be no shakier than the demo.
    dt = env.step_dt
    # The Reference this episode followed (moved by the Object placement in stage 3).
    followed = np.array(reference_q) if reference_q else np.zeros((0, 2, 26))
    result["smoothness"] = {"joints": smoothness(np.array(q), dt), "targets": smoothness(np.array(targets), dt),
                            "reference": smoothness(followed, dt)}
    trace = {"q": np.array(q), "reference_index": np.array(index), "palm_m": palm[:len(q)], "hand_rad": hand[:len(q)],
             "object_pose": np.array(object_pose) if has_object else None,
             "reference_q": np.array(reference_q), "reference_object": np.array(reference_object) if has_object else None,
             "object_joint_q": np.array(object_joint_q) if has_joints else None,
             "reference_object_joint_q": np.array(reference_joint_q) if has_joints else None,
             "cube_pose": np.array(cube_pose) if has_cube else None,
             "reference_cube": np.array(reference_cube) if has_cube else None}
    if hasattr(env, "placement"):
        import math

        entry = int(env.placement[0])
        t = env.bank_translation[entry]
        trace["placement"] = (f"pose: x {100 * float(t[0]):+.1f} cm, y {100 * float(t[1]):+.1f} cm, "
                              f"yaw {math.degrees(float(env.bank_yaw[entry])):+.1f} deg")
    return result, trace


def in_basin(above, distance):
    """The cube's centre above the basin's bottom and within its bottom radius of the basin's axis."""
    from bimanual_dg5fs import scene

    return bool(above > 0.0 and distance < scene.BASIN_BOTTOM_RADIUS)


def cube_scores(env, cube):
    """The cube beside the basin, from its per-step metrics (scored before any reset): its error
    against its track, where it ends, and where it is when the simulated basin is highest."""
    error = np.array(cube["error"])
    lift = int(np.argmax(cube["basin_height"]))
    return {
        "cube_mean_m": float(error.mean()), "cube_max_m": float(error.max()),
        "cube_angle_mean_rad": float(np.mean(cube["angle"])),
        "cube_lift_m": float(max(cube["height"]) - env.reference.cube_pose[0, 2].item()),
        "reference_cube_lift_m": float((env.reference.cube_pose[:, 2].max() - env.reference.cube_pose[0, 2]).item()),
        "cube_final_height_m": cube["height"][-1],
        "cube_final_basin_distance_m": cube["distance"][-1],
        "cube_final_above_basin_m": cube["above"][-1],
        "cube_final_in_basin": in_basin(cube["above"][-1], cube["distance"][-1]),
        "basin_lift_height_m": cube["basin_height"][lift],
        "cube_height_at_lift_m": cube["height"][lift],
        "cube_basin_distance_at_lift_m": cube["distance"][lift],
        # In the basin when the basin is highest, and that at least LIFTED_M above its start.
        "cube_lifted_in_basin": bool(in_basin(cube["above"][lift], cube["distance"][lift]) and
                                     cube["basin_height"][lift] - env.reference.object_pose[0, 2].item() >= LIFTED_M),
    }


def report(result, last_index):
    if result["terminated_at"] is not None:
        print(f"early termination at frame {result['terminated_at']} of {last_index}")
    print(f"frames tracked: {result['frames_tracked']} / {result['frames_to_track']}")
    print(f"Hand-base position error: mean {100 * result['palm_mean_m']:.2f} cm, "
          f"max {100 * result['palm_max_m']:.2f} cm")
    print(f"hand joint RMS error:     mean {result['hand_rms_mean_rad']:.3f} rad, "
          f"max {result['hand_rms_max_rad']:.3f} rad")
    print(f"fingertip error:          mean {100 * result['fingertip_mean_m']:.2f} cm, "
          f"max {100 * result['fingertip_max_m']:.2f} cm")
    if "object_mean_m" in result:
        print(f"Object position error:    mean {100 * result['object_mean_m']:.2f} cm, "
              f"max {100 * result['object_max_m']:.2f} cm; lifted {100 * result['object_lift_m']:.1f} cm "
              f"(Reference {100 * result['reference_lift_m']:.1f} cm)")
    if "cube_mean_m" in result:
        print(f"cube position error:      mean {100 * result['cube_mean_m']:.2f} cm, "
              f"max {100 * result['cube_max_m']:.2f} cm, angle mean {np.degrees(result['cube_angle_mean_rad']):.1f} deg; "
              f"lifted {100 * result['cube_lift_m']:.1f} cm (Reference {100 * result['reference_cube_lift_m']:.1f} cm)")
        print(f"cube at the end:          height {100 * result['cube_final_height_m']:.1f} cm, "
              f"{100 * result['cube_final_basin_distance_m']:.1f} cm from the basin's axis, "
              f"{100 * result['cube_final_above_basin_m']:.1f} cm above its bottom: "
              f"{'in' if result['cube_final_in_basin'] else 'NOT in'} the basin")
        print(f"cube at the basin's lift: basin at {100 * result['basin_lift_height_m']:.1f} cm, cube at "
              f"{100 * result['cube_height_at_lift_m']:.1f} cm, {100 * result['cube_basin_distance_at_lift_m']:.1f} cm "
              f"from its axis: {'lifted in' if result['cube_lifted_in_basin'] else 'NOT lifted in'} the basin")
    if "object_joint_error_mean_rad" in result:
        deg = np.degrees
        print(f"Object joint error:       mean {deg(result['object_joint_error_mean_rad']):.1f} deg, "
              f"max {deg(result['object_joint_error_max_rad']):.1f} deg; final {deg(result['object_joint_final_rad']):.1f} "
              f"deg (Reference {deg(result['reference_joint_final_rad']):.1f}), widest "
              f"{deg(result['object_joint_max_rad']):.1f} deg (Reference {deg(result['reference_joint_max_rad']):.1f})")
    smooth = result.get("smoothness", {})
    if smooth.get("joints") and smooth.get("reference"):
        j, t, r = smooth["joints"], smooth["targets"], smooth["reference"]
        print(f"arm acceleration RMS:     joints {j['arm_acc_rms']:.2f}, targets {t['arm_acc_rms']:.2f}, "
              f"Reference {r['arm_acc_rms']:.2f} rad/s^2")
        print(f"arm power > {HIGH_FREQUENCY_HZ:.0f} Hz:        joints {j['arm_hf']:.1%}, targets {t['arm_hf']:.1%}, "
              f"Reference {r['arm_hf']:.1%}")
        print(f"hand acceleration RMS:    joints {j['hand_acc_rms']:.2f}, targets {t['hand_acc_rms']:.2f}, "
              f"Reference {r['hand_acc_rms']:.2f} rad/s^2")
        print(f"hand power > {HIGH_FREQUENCY_HZ:.0f} Hz:       joints {j['hand_hf']:.1%}, targets {t['hand_hf']:.1%}, "
              f"Reference {r['hand_hf']:.1%}")
    print("SUCCESS" if result["success"] else "not yet")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_arguments(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--replay", action="store_true", help="feed-forward actions from the Reference")
    parser.add_argument("--start", type=int, default=0, help="Reference frame to start from")
    parser.add_argument("--video", default=None, help="write the rollout next to the Reference as an .mp4")
    parser.add_argument("--json", default=None, help="write the scores to this file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override an environment setting as train.py does, e.g. --set object=basin (repeatable)")
    args = parser.parse_args()
    args.num_envs = 1

    env, stack = make_env(args, **parse_settings(args.set), rsi_start_index=args.start)
    try:
        policy = None if args.replay else load_policy(env, args.checkpoint, args.device)
        result, trace = rollout(env, policy, args.start)
        result["checkpoint"] = "replay" if args.replay else args.checkpoint
        report(result, env.reference.last_index)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(result, f, indent=2)
        if args.video:
            from bimanual_training.video import drawn_object, write_rollout_video

            label = "feed-forward replay" if args.replay else os.path.basename(args.checkpoint)
            write_rollout_video(args.video, trace, env.reference, label, result,
                                object_name=drawn_object(env.cfg))
            print(f"video: {args.video}")
    finally:
        env.close()
        stack.close()


if __name__ == "__main__":
    main()
