#!/usr/bin/env python3
"""Builds a Transform bank (CONTEXT.md) for Generalised imitation.

Entry 0 is the Demonstration's own placement; the rest are drawn uniformly at
random inside the training box (seeded) and kept only if feasible by the
Feasibility map's criteria (feasibility_map.check_placements, every frame), until
the bank holds --size entries (at most --max-attempts draws). For each entry the
bank stores both Arms' joints over the whole Reference; Hands, Object track and
Hand-base targets follow from the placement and are computed by the environment.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/build_transform_bank.py \
        --x -0.05 0.05 --y -0.05 0.05 --yaw -15 15 [--size 1536] [--name box]
    # or the box the Feasibility map found:
    ... build_transform_bank.py --from-map training/logs/feasibility/lift_cube_2/summary.json

Writes training/banks/<reference>_<name>.pt and a .json sidecar.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from bimanual_training import REFERENCES_DIR, TRAINING_DIR
from bimanual_training.retarget import PlacementRetargeter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feasibility_map import REASONS, SINGULARITY_SIGMA, _collision_worker_init, check_placements  # noqa: E402

BANKS_DIR = TRAINING_DIR / "banks"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default="lift_cube_2")
    parser.add_argument("--from-map", type=Path, default=None, help="summary.json of a Feasibility map: use its box")
    parser.add_argument("--x", type=float, nargs=2, default=None, help="[m]")
    parser.add_argument("--y", type=float, nargs=2, default=None, help="[m]")
    parser.add_argument("--yaw", type=float, nargs=2, default=None, help="[deg]")
    parser.add_argument("--size", type=int, default=1536)
    parser.add_argument("--max-attempts", type=int, default=None, help="default 4 x size")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", default="box")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--singularity-sigma", type=float, default=None,
                        help="default: the Feasibility map's (or feasibility_map.SINGULARITY_SIGMA)")
    args = parser.parse_args()

    if args.from_map is not None:
        summary = json.loads(args.from_map.read_text())
        box = summary["box"]
        if args.singularity_sigma is None:
            args.singularity_sigma = summary["settings"].get("singularity_sigma")
        args.x, args.y, args.yaw = box["x_m"], box["y_m"], box["yaw_deg"]
    if None in (args.x, args.y, args.yaw):
        raise SystemExit("give --from-map or all of --x, --y, --yaw")
    attempts = args.max_attempts or 4 * args.size
    generator = torch.Generator().manual_seed(args.seed)
    retargeter = PlacementRetargeter(REFERENCES_DIR / f"{args.reference}.npz", device=args.device)
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_collision_worker_init)

    translations, yaws, arm_q = [], [], []
    rejected = {name: 0 for name in REASONS[1:]}
    drawn = 0
    feasible_drawn = 0          # among the random draws (entry 0 excluded)
    started = time.time()
    # Entry 0: the Demonstration's own placement.
    queue = [(torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))]
    while sum(len(y) for y in yaws) < args.size and drawn < attempts:
        if queue:
            translation, yaw = queue.pop()
        else:
            count = min(args.batch, attempts - drawn)
            u = torch.rand(count, 3, generator=generator, dtype=torch.float64)
            translation = torch.stack((args.x[0] + u[:, 0] * (args.x[1] - args.x[0]),
                                       args.y[0] + u[:, 1] * (args.y[1] - args.y[0]),
                                       torch.zeros(count, dtype=torch.float64)), dim=1)
            yaw = torch.deg2rad(args.yaw[0] + u[:, 2] * (args.yaw[1] - args.yaw[0]))
            drawn += count
        out = check_placements(retargeter, translation, yaw, 1, pool, keep_q=True,
                               singularity_sigma=SINGULARITY_SIGMA if args.singularity_sigma is None
                               else args.singularity_sigma)
        keep = out["reason"] == 0
        if not (len(yaw) == 1 and not yaws):
            feasible_drawn += int(keep.sum())
        for code, name in enumerate(REASONS[1:], start=1):
            rejected[name] += int((out["reason"] == code).sum())
        room = args.size - sum(len(y) for y in yaws)
        index = np.flatnonzero(keep)[:room]
        translations.append(translation[index])
        yaws.append(yaw[index])
        arm_q.append(torch.as_tensor(out["arm_q"][index], dtype=torch.float32))
        kept = sum(len(y) for y in yaws)
        print(f"{kept}/{args.size} kept from {drawn} drawn ({time.time() - started:.0f} s); rejected {rejected}",
              flush=True)
    pool.close()
    if not len(yaws) or not len(yaws[0]):
        raise SystemExit("the Demonstration's own placement is not feasible")

    bank = {
        "translation": torch.cat(translations).float(),        # (T, 3) m
        "yaw": torch.cat(yaws).float(),                        # (T,) rad
        "arm_q": torch.cat(arm_q),                             # (T, F, 2, 6), limbs.SIDES order
    }
    kept = len(bank["yaw"])
    acceptance = feasible_drawn / max(drawn, 1)
    BANKS_DIR.mkdir(parents=True, exist_ok=True)
    path = BANKS_DIR / f"{args.reference}_{args.name}.pt"
    torch.save(bank, path)
    sidecar = {"reference": args.reference, "size": kept, "requested": args.size, "drawn": drawn,
               "acceptance": acceptance, "rejected": rejected, "seed": args.seed,
               "x_m": list(args.x), "y_m": list(args.y), "yaw_deg": list(args.yaw),
               "singularity_sigma": SINGULARITY_SIGMA if args.singularity_sigma is None else args.singularity_sigma,
               "frames": int(bank["arm_q"].shape[1]), "entry_0": "the Demonstration's own placement"}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps(sidecar, indent=2))
    if acceptance < 0.9:
        print(f"warning: only {acceptance:.0%} of the box is feasible")
    print(f"written {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
