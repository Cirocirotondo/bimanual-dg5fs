"""Stands up the simulation and the environment (Isaac Lab needs the physics
backend and visualizer chosen before the simulation context exists)."""
import argparse
import contextlib

import warp as wp

# Isaac Lab does not use Warp autodiff; skipping adjoint codegen halves kernel compilation.
wp.config.enable_backward = False


def add_env_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference", default=None, help="name of a Reference in training/references")
    parser.add_argument("--viz", default=None, help="visualizer: newton (kit-less viewer), rerun, viser")


def make_env(args, **overrides):
    """``(env, exit_stack)``; close the stack after ``env.close()``."""
    from isaaclab.app import launch_simulation

    from bimanual_training.motion_imitation_cfg import BimanualMotionImitationEnvCfg
    from bimanual_training.motion_imitation_env import BimanualMotionImitationEnv

    cfg = BimanualMotionImitationEnvCfg()
    cfg.sim.device = args.device
    if args.num_envs is not None:
        cfg.scene.num_envs = int(args.num_envs)
    if args.reference is not None:
        cfg.reference = args.reference
    for name, value in overrides.items():
        setattr(cfg, name, value)
    launcher_args = {
        "device": cfg.sim.device,
        "headless": args.viz is None,
        "visualizer": [args.viz] if args.viz is not None else None,   # a list: the launcher joins it
        "visualizer_explicit": args.viz is not None,
    }
    stack = contextlib.ExitStack()
    stack.enter_context(launch_simulation(cfg, launcher_args))
    try:
        env = BimanualMotionImitationEnv(cfg)
    except Exception:
        stack.close()
        raise
    return env, stack
