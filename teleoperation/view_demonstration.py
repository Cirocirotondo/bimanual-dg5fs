#!/usr/bin/env python3
"""Plays back a Demonstration exactly as recorded: measured Arm and Hand joints of
both Limbs and the Object pose, at the recorded timing (no physics, no IK).

    env -u PYTHONPATH uv run python -m teleoperation.view_demonstration <demo .npz or name>
    ... --speed 0.5     # slow motion
    ... --loop          # repeat
    ... --commanded     # show the commanded joints instead of the measured ones

To run the recorded inputs through the whole pipeline again instead, use
``python -m teleoperation.run_sim --source replay --replay <demo .npz>``.
"""
import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from bimanual_dg5fs import scene
from teleoperation.core import TELEOP_DIR
from teleoperation.run_sim import InsetView, set_camera


def find(name):
    path = Path(name)
    for candidate in (path, path.with_suffix(".npz"), TELEOP_DIR / "demonstrations" / path.name,
                      (TELEOP_DIR / "demonstrations" / path.name).with_suffix(".npz")):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Demonstration not found: {name}")


class Playback:
    def __init__(self, path, commanded=False):
        self.data_file = np.load(path)
        self.time = self.data_file["time"] - self.data_file["time"][0]
        self.model = scene.build_model()
        self.data = mujoco.MjData(self.model)
        kind = "commanded" if commanded else "measured"
        self.joints = []                 # (qpos addresses, recorded array)
        for side in scene.SIDES:
            self.joints.append(([self.model.joint(f"{side}/{j}").qposadr[0] for j in scene.ARM_JOINTS],
                                self.data_file[f"{side}_arm_q_{kind}"]))
            self.joints.append(([self.model.joint(f"{side}/{j}").qposadr[0] for j in scene.HAND_JOINTS],
                                self.data_file[f"{side}_hand_q_{kind}"]))
        self.object_adr = self.model.joint("object_free").qposadr[0]
        self.object = self.data_file["object_pose"]          # x y z qx qy qz qw
        scene.reset_to_home(self.model, self.data)

    def show(self, i):
        for addresses, values in self.joints:
            self.data.qpos[addresses] = values[i]
        x, y, z, qx, qy, qz, qw = self.object[i]
        self.data.qpos[self.object_adr:self.object_adr + 7] = [x, y, z, qw, qx, qy, qz]
        mujoco.mj_forward(self.model, self.data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("demonstration", help="path or name of a .npz in teleoperation/demonstrations")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--commanded", action="store_true", help="commanded instead of measured joints")
    parser.add_argument("--no-inset", action="store_true", help="no horizontal view in the corner")
    args = parser.parse_args(argv)

    path = find(args.demonstration)
    info_path = path.with_suffix(".json")
    if info_path.exists():
        info = json.loads(info_path.read_text())
        print(f"{path.name}: {info['samples']} samples, {info['duration_s']:.1f} s, success {info['success']}")
    playback = Playback(path, args.commanded)
    inset = None if args.no_inset else InsetView(playback.model)
    with mujoco.viewer.launch_passive(playback.model, playback.data) as viewer:
        set_camera(viewer.cam)
        while viewer.is_running():
            start = time.monotonic()
            for i, t in enumerate(playback.time):
                if not viewer.is_running():
                    return
                playback.show(i)
                if inset is not None:
                    inset.draw(viewer, playback.data)
                viewer.sync()
                time.sleep(max(0.0, start + t / args.speed - time.monotonic()))
            if not args.loop:
                time.sleep(1.0)
                break


if __name__ == "__main__":
    main()
