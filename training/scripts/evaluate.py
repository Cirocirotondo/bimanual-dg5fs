#!/usr/bin/env python3
"""Runs the Reference once from its first frame and scores the tracking.

    # a trained policy (deterministic: the mean action)
    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/evaluate.py \
        --checkpoint training/logs/motion_imitation/<run>/model_3000.pt [--viz newton]
    # no policy: feed-forward actions from the Reference (checks the task is feasible)
    ... training/scripts/evaluate.py --replay

Success (Motion imitation stage): the whole Reference without early termination,
mean Hand-base position error under 2 cm and mean hand joint RMS error under
0.1 rad (worst Limb at each step).
"""
import argparse

import torch

from bimanual_training.launch import add_env_arguments, make_env

PALM_SUCCESS_M = 0.02
HAND_SUCCESS_RAD = 0.1


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_arguments(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--replay", action="store_true", help="feed-forward actions from the Reference")
    parser.add_argument("--start", type=int, default=0, help="Reference frame to start from")
    args = parser.parse_args()
    args.num_envs = 1

    env, stack = make_env(args, rsi_start_index=args.start)
    try:
        policy = None if args.replay else load_policy(env, args.checkpoint, args.device)
        observations, _ = env.reset()
        palm, hand, tips = [], [], []
        terminated = False
        with torch.inference_mode():
            while True:
                if policy is None:
                    actions = env.reference_actions()
                else:
                    from tensordict import TensorDict
                    actions = policy(TensorDict(observations, batch_size=[env.num_envs]))
                index = int(env.reference_index[0])
                observations, _, early, timeout, extras = env.step(actions)
                metrics = extras["metrics"]
                palm.append(float(metrics["palm_position_error_m"][0]))
                hand.append(float(metrics["hand_rms_error_rad"][0]))
                tips.append(float(metrics["fingertip_error_m"][0]))
                if bool(early[0]):
                    terminated = True
                    print(f"early termination at frame {index + 1} of {env.reference.last_index}")
                    break
                if bool(timeout[0]):
                    break
        palm_t, hand_t, tips_t = torch.tensor(palm), torch.tensor(hand), torch.tensor(tips)
        print(f"frames tracked: {len(palm)} / {env.reference.last_index - args.start}")
        print(f"Hand-base position error: mean {100 * palm_t.mean():.2f} cm, max {100 * palm_t.max():.2f} cm")
        print(f"hand joint RMS error:     mean {hand_t.mean():.3f} rad, max {hand_t.max():.3f} rad")
        print(f"fingertip error:          mean {100 * tips_t.mean():.2f} cm, max {100 * tips_t.max():.2f} cm")
        success = (not terminated and palm_t.mean() < PALM_SUCCESS_M and hand_t.mean() < HAND_SUCCESS_RAD)
        print("SUCCESS" if success else "not yet")
    finally:
        env.close()
        stack.close()


if __name__ == "__main__":
    main()
