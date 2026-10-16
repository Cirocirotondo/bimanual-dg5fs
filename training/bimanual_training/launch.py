"""Stands up the simulation and the environment (Isaac Lab needs the physics
backend and visualizer chosen before the simulation context exists)."""
import argparse
import contextlib
import os

# OpenUSD before usd-core 26.5 can corrupt the heap when it parses physics in parallel
# (Isaac Lab's own Newton tests run with this limit). Newton parses env_0 once at clone
# time; with a rigid body of more than ~10 colliders (the basin has 55) it segfaults at
# random. Read when pxr is first imported, so this module must be imported before it.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import warp as wp  # noqa: E402

# Isaac Lab does not use Warp autodiff; skipping adjoint codegen halves kernel compilation.
wp.config.enable_backward = False


def add_env_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference", default=None, help="name of a Reference in training/references")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1,
                        help="Training stage: 1 Motion imitation, 2 Fixed-object imitation (with the Object the "
                             "Reference was demonstrated with, or --set object=...), "
                             "3 Generalised imitation (cube placed anywhere a Transform bank covers)")
    parser.add_argument("--viz", default=None, help="visualizer: newton (kit-less viewer), rerun, viser")


def parse_settings(pairs):
    """``KEY=VALUE`` strings -> {key: value}, each value cast to the type of the
    environment setting it replaces (unknown keys are an error, not a silent no-op)."""
    from bimanual_training.motion_imitation_cfg import BimanualPlacementImitationEnvCfg

    defaults = BimanualPlacementImitationEnvCfg()   # stage 3's settings include stages 1 and 2's
    settings = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not hasattr(defaults, key):
            raise SystemExit(f"--set {pair}: no environment setting {key!r}")
        current = getattr(defaults, key)
        if isinstance(current, bool):
            settings[key] = value.lower() in ("1", "true", "yes")
        elif current is None:
            # Optional settings: a number, or a name (e.g. object=basin).
            try:
                settings[key] = None if value.lower() == "none" else float(value)
            except ValueError:
                settings[key] = value
        else:
            settings[key] = type(current)(value)
    return settings


def make_env(args, **overrides):
    """``(env, exit_stack)``; close the stack after ``env.close()``."""
    from isaaclab.app import launch_simulation

    if getattr(args, "stage", 1) == 3:
        from bimanual_training.motion_imitation_cfg import BimanualPlacementImitationEnvCfg as Cfg
        from bimanual_training.placement_imitation_env import BimanualPlacementImitationEnv as Env
    elif getattr(args, "stage", 1) == 2:
        from bimanual_training.motion_imitation_cfg import BimanualObjectImitationEnvCfg as Cfg
        from bimanual_training.object_imitation_env import BimanualObjectImitationEnv as Env
    else:
        from bimanual_training.motion_imitation_cfg import BimanualMotionImitationEnvCfg as Cfg
        from bimanual_training.motion_imitation_env import BimanualMotionImitationEnv as Env

    cfg = Cfg()
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
        env = Env(cfg)
    except Exception:
        stack.close()
        raise
    return env, stack
