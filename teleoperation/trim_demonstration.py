#!/usr/bin/env python3
"""Trims a Demonstration and saves it as a new one (the original is kept).

    # keep only the part where both Limbs are engaged (drops engaging / disengaging)
    env -u PYTHONPATH uv run python -m teleoperation.trim_demonstration <demo> --engaged
    # or explicit times, in seconds from the start of the recording
    env -u PYTHONPATH uv run python -m teleoperation.trim_demonstration <demo> --start 1.5 --end 20

Writes <demo>_trimmed.npz / .json next to the original (--suffix to change).
"""
import argparse
import json

import numpy as np

from teleoperation.inputs import SIDES
from teleoperation.view_demonstration import find


def engaged_window(data):
    """First and last sample index where every Limb is engaged."""
    both = np.all([data[f"{side}_engaged"].astype(bool) for side in SIDES], axis=0)
    if not both.any():
        raise SystemExit("no sample with both Limbs engaged")
    index = np.flatnonzero(both)
    return int(index[0]), int(index[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("demonstration")
    parser.add_argument("--engaged", action="store_true", help="keep the part where both Limbs are engaged")
    parser.add_argument("--start", type=float, help="start time [s] from the beginning of the recording")
    parser.add_argument("--end", type=float, help="end time [s] from the beginning of the recording")
    parser.add_argument("--suffix", default="_trimmed")
    args = parser.parse_args(argv)

    path = find(args.demonstration)
    data = np.load(path)
    t = data["time"] - data["time"][0]
    first, last = engaged_window(data) if args.engaged else (0, len(t) - 1)
    if args.start is not None:
        first = max(first, int(np.searchsorted(t, args.start)))
    if args.end is not None:
        last = min(last, int(np.searchsorted(t, args.end, side="right")) - 1)
    if last <= first:
        raise SystemExit("empty selection")

    out = path.with_name(path.stem + args.suffix + ".npz")
    np.savez_compressed(out, **{k: data[k][first:last + 1] for k in data.files})
    info_path = path.with_suffix(".json")
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.update({
        "samples": last - first + 1,
        "duration_s": float(t[last] - t[first]),
        "trimmed_from": path.name,
        "trim_s": [float(t[first]), float(t[last])],
    })
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"{path.name}: kept {t[first]:.2f} - {t[last]:.2f} s of {t[-1]:.2f} s "
          f"({last - first + 1} samples) -> {out.name}")


if __name__ == "__main__":
    main()
