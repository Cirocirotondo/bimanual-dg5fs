#!/usr/bin/env python3
"""Expands a Training-stage-1 checkpoint to stage 2's observations.

Stage 2 appends the Object's observations after stage 1's, so the policy and the
critic keep every weight and get zero weights for the new inputs: at the first
iteration they act exactly as the stage-1 policy, and learn to use the Object
from there. The observation normalizers treat the new inputs as already
normalised (mean 0, std 1) until the running statistics catch up.

The result has no optimizer state (Adam starts afresh; train.py --resume then
loads it without one) and iteration 0, so the stage-2 run counts from 0.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/expand_checkpoint.py \
        <stage-1 model_N.pt> <out>/model_0.pt [--object book | --observations 223]

The size is stage 2's with the given Object: 223 for the cube and the basin, 231
for the book (its hinge and cover lip are appended after the rigid Objects' layout),
255 for bag and bag_sealed (10 joints and both short ends), 241 for basin_cube.
"""
import argparse
from pathlib import Path

import torch


def expand(state, new_dim):
    old_dim = state["mlp.0.weight"].shape[1]
    extra = new_dim - old_dim
    if extra < 0:
        raise SystemExit(f"cannot shrink {old_dim} observations to {new_dim}")
    weight = state["mlp.0.weight"]
    state["mlp.0.weight"] = torch.cat((weight, weight.new_zeros(weight.shape[0], extra)), dim=1)
    for key, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
        if key in state:
            value = state[key]
            state[key] = torch.cat((value, value.new_full((*value.shape[:-1], extra), fill)), dim=-1)
    return old_dim


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--observations", type=int, default=None, help="stage 2's size (default: its config)")
    parser.add_argument("--object", default="cube", help="stage 2's Object, which sets the size (default cube)")
    args = parser.parse_args()

    if args.observations is None:
        from bimanual_training.motion_imitation_cfg import object_observation_space

        args.observations = object_observation_space(args.object)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old = expand(checkpoint["actor_state_dict"], args.observations)
    expand(checkpoint["critic_state_dict"], args.observations)
    checkpoint["optimizer_state_dict"] = None
    checkpoint["iter"] = 0
    checkpoint["infos"] = {"expanded_from": str(args.checkpoint), "observations": [old, args.observations]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"{args.checkpoint} ({old} observations) -> {args.output} ({args.observations})")


if __name__ == "__main__":
    main()
