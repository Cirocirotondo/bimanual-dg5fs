#!/usr/bin/env python3
"""Evaluates a run's checkpoint from several start frames, with videos.

For each run directory: its newest checkpoint (or --checkpoint NAME), --repeats
deterministic rollouts per start frame (scripts/evaluate.py's rollout), a video
of the one with the median palm error next to the Reference, and a summary, all
in <run dir>/eval/<checkpoint stem>/:

    start_<N>.json, start_<N>.mp4, summary.md

Repeats, because the GPU physics is not bit-reproducible and a closed-loop
rollout amplifies that: the same checkpoint from frame 0 measured 2.01 and
2.69 cm mean palm error in two runs of the same command.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/evaluate_run.py \
        training/logs/motion_imitation/<run> [<run> ...] [--starts 0 300 500 700] [--checkpoint model_5999.pt]

The environment is built once and reused for every run and start frame.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")       # before anything imports mujoco

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_policy, rollout  # noqa: E402

from bimanual_training.launch import add_env_arguments, make_env  # noqa: E402


def short_name(run):
    """The run name without its launch timestamp (the video caption has little room);
    a pulled run keeps its host: TARS_2026-10-12_15-24-54_m1_adaptive -> TARS m1_adaptive."""
    return re.sub(r"^(?:([A-Z]+)_)?\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_",
                  lambda m: f"{m.group(1)} " if m.group(1) else "", run)


def run_reference(run_dir):
    config = run_dir / "config.json"
    if config.is_file():
        return json.loads(config.read_text())["env"].get("reference", "lift_cube")
    return "lift_cube"


def spread_placements(env, count):
    """Bank entry 0 (the Demonstration's placement) and ``count`` entries spread over the
    bank by farthest-point sampling (x, y in metres, yaw through a 0.1 m lever arm)."""
    import torch

    from bimanual_training.placement import NEAREST_YAW_LEVER_ARM_M, fold_yaw

    points = torch.cat((env.bank_translation[:, :2],
                        (NEAREST_YAW_LEVER_ARM_M * fold_yaw(env.bank_yaw)).unsqueeze(1)), dim=1)
    chosen = [0]
    distance = (points - points[0]).norm(dim=1)
    for _ in range(min(count, len(points) - 1)):
        index = int(distance.argmax())
        chosen.append(index)
        distance = torch.minimum(distance, (points - points[index]).norm(dim=1))
    return chosen


def placement_label(env, index):
    import math

    t, y = env.bank_translation[index], float(env.bank_yaw[index])
    return f"x {100 * float(t[0]):+.1f} cm, y {100 * float(t[1]):+.1f} cm, yaw {math.degrees(y):+.1f}°"


# The early-termination settings a run was trained with: the evaluation ends an episode
# where training would have, not on the cfg defaults (a run trained with a 20 cm palm
# threshold would otherwise be cut at 8 cm).
TERMINATION_SETTINGS = ("palm_keypoint_threshold_m", "hand_position_threshold_rad", "grace_steps",
                        "object_position_threshold_m", "cube_position_threshold_m", "object_joint_threshold_rad")


def run_terminations(run_dir):
    """``{setting: value}`` of TERMINATION_SETTINGS in the run's config.json (empty for older runs)."""
    config = run_dir / "config.json"
    env = json.loads(config.read_text())["env"] if config.is_file() else {}
    return {key: env[key] for key in TERMINATION_SETTINGS if key in env}


def run_object(run_dir):
    """The run's Object (stage 2), None when its config predates the choice (the
    environment then takes the Reference's)."""
    config = run_dir / "config.json"
    if config.is_file():
        return json.loads(config.read_text())["env"].get("object")
    return None


def run_stage(run_dir):
    config = run_dir / "config.json"
    if config.is_file():
        return int(json.loads(config.read_text())["env"].get("stage", 1))
    return 1


def newest_checkpoint(run_dir):
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not models:
        raise SystemExit(f"no checkpoint in {run_dir}")
    return models[-1]


def ratio(repeats, key):
    """Median over the repeats of (targets' value) / (Reference's value) for one smoothness key."""
    values = []
    for r in repeats:
        smooth = r.get("smoothness", {})
        target, reference = smooth.get("targets", {}).get(key), smooth.get("reference", {}).get(key)
        if target is not None and reference:
            values.append(target / reference)
    return float(np.median(values)) if values else float("nan")


def aggregate(start, repeats):
    """Median and range over the repeats of one start frame."""
    def stat(key, scale=1.0):
        values = np.array([r[key] for r in repeats]) * scale
        return float(np.median(values)), float(values.min()), float(values.max())

    return {
        "start": start,
        "repeats": len(repeats),
        "successes": sum(r["success"] for r in repeats),
        "terminations": sum(r["terminated_at"] is not None for r in repeats),
        "frames_tracked_min": min(r["frames_tracked"] for r in repeats),
        "frames_to_track": repeats[0]["frames_to_track"],
        "palm_mean_cm": stat("palm_mean_m", 100.0),
        "hand_rms_mean_rad": stat("hand_rms_mean_rad"),
        "fingertip_mean_cm": stat("fingertip_mean_m", 100.0),
        "object_mean_cm": stat("object_mean_m", 100.0) if "object_mean_m" in repeats[0] else None,
        "object_max_cm": stat("object_max_m", 100.0) if "object_mean_m" in repeats[0] else None,
        "object_lift_cm": stat("object_lift_m", 100.0) if "object_mean_m" in repeats[0] else None,
        # An articulated Object (the book's hinge) [deg].
        **({"object_joint_error_mean_deg": stat("object_joint_error_mean_rad", 180.0 / np.pi),
            "object_joint_error_max_deg": stat("object_joint_error_max_rad", 180.0 / np.pi),
            "object_joint_final_deg": stat("object_joint_final_rad", 180.0 / np.pi),
            "reference_joint_final_deg": stat("reference_joint_final_rad", 180.0 / np.pi)}
           if "object_joint_error_mean_rad" in repeats[0] else {}),
        # The cube beside the basin (basin_cube): its error, final height, and in how many
        # repeats it ends in the basin / is lifted in it (evaluate.py's cube_lifted_in_basin).
        **({"cube_mean_cm": stat("cube_mean_m", 100.0), "cube_max_cm": stat("cube_max_m", 100.0),
            "cube_final_height_cm": stat("cube_final_height_m", 100.0),
            "cube_final_in_basin": sum(r["cube_final_in_basin"] for r in repeats),
            "cube_lifted_in_basin": sum(r["cube_lifted_in_basin"] for r in repeats)}
           if "cube_mean_m" in repeats[0] else {}),
        # Shakiness of the commands sent to the drives, relative to the Reference's own.
        "arm_target_acc_ratio": ratio(repeats, "arm_acc_rms"),
        "hand_target_acc_ratio": ratio(repeats, "hand_acc_rms"),
        "arm_target_hf": float(np.median([r["smoothness"]["targets"].get("arm_hf", np.nan) for r in repeats])),
        "hand_target_hf": float(np.median([r["smoothness"]["targets"].get("hand_hf", np.nan) for r in repeats])),
        "rollouts": repeats,
    }


def summary_table(run_dir, checkpoint, aggregates):
    lines = [f"# {run_dir.name} / {checkpoint.name}", "",
             "Deterministic rollouts (mean action), worst Limb per frame, repeated because the GPU",
             "physics is not bit-reproducible. Success: the whole Reference without early termination,",
             "palm < 2 cm, hand RMS < 0.1 rad (means). Cells: median [min - max] over the repeats.", "",
             "Smoothness: RMS acceleration of the targets sent to the drives over the Reference's",
             "(x1 = as smooth as the demo), and their share of power above 4 Hz.", "",
             "| start | successes | early terminations | palm mean [cm] | hand RMS mean [rad] | tips mean [cm] "
             "| arm accel. xRef | hand accel. xRef | arm / hand > 4 Hz | video |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def video(a):
        return a.get("video", "start_{}.mp4".format(a["start"]))

    def cell(triple, fmt):
        return f"{triple[0]:{fmt}} [{triple[1]:{fmt}} - {triple[2]:{fmt}}]"

    for a in aggregates:
        lines.append(
            f"| {a.get('placement', a['start'])} | {a['successes']}/{a['repeats']} | {a['terminations']}/{a['repeats']} | "
            f"{cell(a['palm_mean_cm'], '.2f')} | {cell(a['hand_rms_mean_rad'], '.3f')} | "
            f"{cell(a['fingertip_mean_cm'], '.2f')} | {a['arm_target_acc_ratio']:.2f} | "
            f"{a['hand_target_acc_ratio']:.2f} | {a['arm_target_hf']:.1%} / {a['hand_target_hf']:.1%} | "
            f"[{video(a)}]({video(a)}) |")
        if a.get("object_mean_cm"):
            lines.append(f"|  | Object: mean {cell(a['object_mean_cm'], '.1f')} cm, max {cell(a['object_max_cm'], '.1f')} cm, "
                         f"lifted {cell(a['object_lift_cm'], '.1f')} cm |  |  |  |  |  |  |  |  |")
        if a.get("cube_mean_cm"):
            lines.append(f"|  | cube: mean {cell(a['cube_mean_cm'], '.1f')} cm, max {cell(a['cube_max_cm'], '.1f')} cm, "
                         f"final height {cell(a['cube_final_height_cm'], '.1f')} cm; in the basin at the end "
                         f"{a['cube_final_in_basin']}/{a['repeats']}, lifted in it {a['cube_lifted_in_basin']}/"
                         f"{a['repeats']} |  |  |  |  |  |  |  |  |")
        if a.get("object_joint_error_mean_deg"):
            lines.append(f"|  | hinge error: mean {cell(a['object_joint_error_mean_deg'], '.1f')} deg, max "
                         f"{cell(a['object_joint_error_max_deg'], '.1f')} deg; final {cell(a['object_joint_final_deg'], '.0f')} "
                         f"deg (Reference {cell(a['reference_joint_final_deg'], '.0f')}) |  |  |  |  |  |  |  |  |")
    return "\n".join(lines) + "\n"


def evaluate_start(env, policy, run_dir, checkpoint, out, args, repeats, start, stem=None, label=None):
    runs = [rollout(env, policy, start) for _ in range(repeats)]
    results = [r for r, _ in runs]
    summary = aggregate(start, results)
    summary["checkpoint"] = str(checkpoint)
    if label:
        summary["placement"] = label
    stem = stem or f"start_{start}"
    summary["video"] = f"{stem}.mp4"
    print(f"-- {label + ', ' if label else ''}start {start}: {summary['successes']}/{repeats} successes, "
          f"{summary['terminations']}/{repeats} early terminations, palm mean median "
          f"{summary['palm_mean_cm'][0]:.2f} cm [{summary['palm_mean_cm'][1]:.2f} - "
          f"{summary['palm_mean_cm'][2]:.2f}], hand RMS median {summary['hand_rms_mean_rad'][0]:.3f} rad, "
          f"target accel. x{summary['arm_target_acc_ratio']:.2f} arm / x{summary['hand_target_acc_ratio']:.2f} hand"
          + (f", hinge error mean median {summary['object_joint_error_mean_deg'][0]:.1f} deg, final "
             f"{summary['object_joint_final_deg'][0]:.0f} deg (Reference {summary['reference_joint_final_deg'][0]:.0f})"
             if "object_joint_error_mean_deg" in summary else "")
          + (f", cube mean median {summary['cube_mean_cm'][0]:.1f} cm, in the basin at the end "
             f"{summary['cube_final_in_basin']}/{repeats}" if "cube_mean_cm" in summary else ""),
          flush=True)
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    if not args.no_video:
        from bimanual_training.video import drawn_object, write_rollout_video

        # The rollout with the median palm error stands for the start frame.
        order = np.argsort([r["palm_mean_m"] for r in results])
        result, trace = runs[int(order[len(order) // 2])]
        write_rollout_video(str(out / f"{stem}.mp4"), trace, env.reference,
                            f"{short_name(run_dir.name)} / {checkpoint.stem}", result,
                            object_name=drawn_object(env.cfg))
    return summary


def evaluate_placements(env, policy, run_dir, checkpoint, out, args, repeats):
    """Stage 3: from frame 0 at the Demonstration's placement and at spread bank entries.

    One p<entry>_start_0.json / .mp4 per placement, and start_0.json pooling every
    placement (what the wave table reads)."""
    per_placement = []
    for index in spread_placements(env, args.placements):
        env.cfg.placement_index = index
        label = placement_label(env, index)
        per_placement.append(evaluate_start(env, policy, run_dir, checkpoint, out, args, repeats, 0,
                                            stem=f"p{index}_start_0", label=f"bank {index} ({label})"))
    pooled = aggregate(0, [r for summary in per_placement for r in summary["rollouts"]])
    pooled["checkpoint"] = str(checkpoint)
    pooled["placements"] = [s["placement"] for s in per_placement]
    (out / "start_0.json").write_text(json.dumps(pooled, indent=2))
    demo_video = out / "p0_start_0.mp4"
    if demo_video.exists():
        import shutil

        shutil.copyfile(demo_video, out / "start_0.mp4")
    return per_placement


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_arguments(parser)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--starts", type=int, nargs="+", default=[0, 300, 500, 700])
    parser.add_argument("--checkpoint", default=None, help="checkpoint file name inside each run (default: newest)")
    parser.add_argument("--repeats", type=int, default=None,
                        help="rollouts per start frame (default 5; 3 per placement in stage 3)")
    parser.add_argument("--placements", type=int, default=6,
                        help="stage 3: bank entries spread over the bank, besides the Demonstration's")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    args.num_envs = 1
    if args.reference is None:
        # The Reference the runs were trained on (all runs of one call must share it).
        references = {run_reference(run) for run in args.runs}
        if len(references) > 1:
            raise SystemExit(f"runs trained on different References {references}: evaluate them separately")
        args.reference = references.pop()
    stages = {run_stage(run) for run in args.runs}
    if len(stages) > 1:
        raise SystemExit(f"runs of different Training stages {stages}: evaluate them separately")
    args.stage = stages.pop()

    terminations = {json.dumps(run_terminations(run), sort_keys=True) for run in args.runs}
    if len(terminations) > 1:
        raise SystemExit(f"runs with different termination settings {terminations}: evaluate them separately")
    terminations = json.loads(terminations.pop())
    print(f"termination settings as trained: {terminations}", flush=True)
    if args.stage == 3:
        settings = {(cfg.get("bank"), bool(cfg.get("observe_placement", False)), bool(cfg.get("canonical_frame", False)))
                    for cfg in (json.loads((run / "config.json").read_text())["env"] for run in args.runs)}
        if len(settings) > 1:
            raise SystemExit(f"runs with different banks / observations {settings}: evaluate them separately")
        bank, observe_placement, canonical_frame = settings.pop()
        env, stack = make_env(args, rsi_start_index=0, bank=bank, placement_index=0,
                              observe_placement=observe_placement, canonical_frame=canonical_frame, **terminations)
    else:
        objects = {run_object(run) for run in args.runs}
        if len(objects) > 1:
            raise SystemExit(f"runs with different Objects {objects}: evaluate them separately")
        settings = {"object": objects.pop()} if args.stage == 2 else {}
        env, stack = make_env(args, rsi_start_index=0, **settings, **terminations)
    repeats = args.repeats or (3 if args.stage == 3 else 5)
    try:
        for run_dir in args.runs:
            checkpoint = run_dir / args.checkpoint if args.checkpoint else newest_checkpoint(run_dir)
            out = run_dir / "eval" / checkpoint.stem
            out.mkdir(parents=True, exist_ok=True)
            print(f"== {run_dir.name} / {checkpoint.name}", flush=True)
            policy = load_policy(env, str(checkpoint), args.device)
            aggregates = []
            if args.stage == 3:
                aggregates = evaluate_placements(env, policy, run_dir, checkpoint, out, args, repeats)
            else:
                for start in args.starts:
                    aggregates.append(evaluate_start(env, policy, run_dir, checkpoint, out, args, repeats, start))
            (out / "summary.md").write_text(summary_table(run_dir, checkpoint, aggregates))
            print(f"results: {out}", flush=True)
    finally:
        env.close()
        stack.close()


if __name__ == "__main__":
    main()
