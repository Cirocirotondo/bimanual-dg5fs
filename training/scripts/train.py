#!/usr/bin/env python3
"""Trains the Motion imitation policy with RSL-RL PPO.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/train.py [--num-envs 1024]
    ... --max-iterations 5 --num-envs 64          # smoke test
    ... --resume training/logs/motion_imitation/<run>/model_1000.pt   # --max-iterations more
    ... --schedule adaptive --learning-rate 1e-3                      # RSL-RL's adaptive learning rate

Logs and checkpoints go to training/logs/motion_imitation/<date>_<time>[_name]
(TensorBoard: tensorboard --logdir training/logs).
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from bimanual_training import LOGS_DIR
from bimanual_training.launch import add_env_arguments, make_env, parse_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_arguments(parser)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="", help="suffix of the run directory")
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    parser.add_argument("--log-dir", default=None, help="exact run directory (the fleet supervisor sets it)")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--schedule", choices=("fixed", "adaptive"), default=None)
    parser.add_argument("--init-std", type=float, default=None,
                        help="initial exploration noise of the policy (default 1.0)")
    parser.add_argument("--entropy-coef", type=float, default=None, help="entropy bonus (default 0.001)")
    parser.add_argument("--min-std", type=float, default=None,
                        help="floor of the exploration noise: low-noise runs collapse once it falls below ~0.13")
    parser.add_argument("--rsi-start-probability", type=float, default=None,
                        help="fraction of episodes starting at the Reference's first frame")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override an environment setting, e.g. --set arm_action_rate_weight=0.4 (repeatable)")
    args = parser.parse_args()

    overrides = {"seed": args.seed}
    if args.rsi_start_probability is not None:
        overrides["rsi_start_probability"] = args.rsi_start_probability
    overrides.update(parse_settings(args.set))
    env, stack = make_env(args, **overrides)
    try:
        import importlib.metadata as metadata

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner

        from bimanual_training.agents import MotionImitationPPORunnerCfg

        agent_cfg = MotionImitationPPORunnerCfg()
        agent_cfg.seed = args.seed
        agent_cfg.device = args.device
        if args.max_iterations is not None:
            agent_cfg.max_iterations = args.max_iterations
        if args.learning_rate is not None:
            agent_cfg.algorithm.learning_rate = args.learning_rate
        if args.schedule is not None:
            agent_cfg.algorithm.schedule = args.schedule
        if args.init_std is not None:
            agent_cfg.actor.distribution_cfg.init_std = args.init_std
        if args.entropy_coef is not None:
            agent_cfg.algorithm.entropy_coef = args.entropy_coef
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        if args.log_dir:
            log_dir = Path(args.log_dir).resolve()
        else:
            run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args.name}" if args.name else "")
            log_dir = LOGS_DIR / agent_cfg.experiment_name / run
        os.makedirs(log_dir, exist_ok=True)
        (log_dir / "config.json").write_text(json.dumps(
            {"env": env.cfg.to_dict(), "agent": agent_cfg.to_dict(), "args": vars(args)}, indent=2, default=str))

        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        train_cfg = agent_cfg.to_dict()
        if args.min_std is not None:
            # RSL-RL's GaussianDistribution clamps its std to std_range; Isaac Lab's cfg
            # does not expose it, so it is set in the dict the runner is built from.
            train_cfg["actor"]["distribution_cfg"]["std_range"] = [args.min_std, 1.0e6]
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=str(log_dir), device=agent_cfg.device)
        if args.resume:
            import torch

            # A checkpoint expanded to a new observation layout (scripts/expand_checkpoint.py)
            # carries no optimizer state: Adam starts afresh.
            fresh = torch.load(args.resume, map_location="cpu", weights_only=False).get("optimizer_state_dict") is None
            runner.load(args.resume, load_cfg={"actor": True, "critic": True, "optimizer": not fresh,
                                               "iteration": True} if fresh else None)
            if args.init_std is not None:
                # On a resume the checkpoint's noise would win: set it explicitly
                # (e.g. lowering it to fine-tune a policy for smoothness).
                with torch.no_grad():
                    runner.alg._raw_actor.distribution.std_param.fill_(args.init_std)
        print(f"[train] {env.num_envs} envs, Reference '{env.cfg.reference}' ({env.reference.frames} frames), "
              f"lr {agent_cfg.algorithm.learning_rate} {agent_cfg.algorithm.schedule}, "
              f"init std {agent_cfg.actor.distribution_cfg.init_std}, entropy {agent_cfg.algorithm.entropy_coef}, "
              f"RSI {env.cfg.rsi_start_probability:.0%} at frame 0, "
              f"from iteration {runner.current_learning_iteration}, logging to {log_dir}", flush=True)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    finally:
        env.close()
        stack.close()


if __name__ == "__main__":
    main()
