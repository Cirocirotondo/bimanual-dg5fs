#!/usr/bin/env python3
"""Bimanual teleoperation of the Scene with two MANUS Gloves and one camera.

    env -u PYTHONPATH uv run python -m teleoperation.run_sim                  # live Gloves + camera
    env -u PYTHONPATH uv run python -m teleoperation.run_sim --source mock    # no hardware
    env -u PYTHONPATH uv run python -m teleoperation.run_sim --source replay --replay <demo.npz>
    ... --record                                                               # save Demonstrations

Commands (type in this terminal, then Enter):
    c      engage / clutch both Limbs      lc / rc   engage / clutch the left / right Limb
    h      home: clutch both, Arms back to the Home pose, Hands open
    r      reset the Scene (discards the episode being recorded)
    ok     save the episode as a success and reset   (with --record)
    fail   save the episode as a failure and reset   (with --record)
    q      quit (an unfinished episode is saved with success unknown)
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from bimanual_dg5fs import scene
from teleoperation.core import TELEOP_DIR, TeleopCore, load_config
from teleoperation.finger_retargeting import joint_limits_from_model
from teleoperation.inputs import SIDES, LiveSource, MockSource, ReplaySource
from teleoperation.outputs import SimOutput
from teleoperation.recorder import DemonstrationRecorder


def make_source(args, config):
    if args.source == "mock":
        return MockSource()
    if args.source == "replay":
        if not args.replay:
            sys.exit("--source replay needs --replay <demonstration.npz>")
        return ReplaySource(args.replay, loop=args.loop)
    i = config["inputs"]
    return LiveSource(i["glove_address"], i["pose_address"], i["wrist_board_index"],
                      i.get("skeleton_address"), i.get("glove_ids"))


def start_command_reader():
    commands = queue.Queue()

    def read():
        for line in sys.stdin:
            commands.put(line.strip().lower())

    threading.Thread(target=read, daemon=True).start()
    return commands


def set_camera(cam):
    """Main view, from above: the Operator's left on the left, the Hands pointing up."""
    cam.lookat[:] = (0.0, -0.40, 0.15)
    cam.distance = 2.2
    cam.azimuth = -90.0
    cam.elevation = -89.0


def set_inset_camera(cam):
    """Horizontal view from in front of the Hands, at table height, towards the Arms."""
    cam.lookat[:] = (0.0, -0.45, 0.25)
    cam.distance = 1.4
    cam.azimuth = 90.0
    cam.elevation = -8.0


class InsetView:
    """A second, fixed view drawn in the bottom-right corner of the viewer window."""

    WIDTH, HEIGHT, MARGIN = 480, 360, 10

    def __init__(self, model):
        self.renderer = mujoco.Renderer(model, self.HEIGHT, self.WIDTH)
        self.camera = mujoco.MjvCamera()
        set_inset_camera(self.camera)

    def draw(self, viewer, data):
        window = viewer.viewport
        if window is None or window.width < self.WIDTH + 2 * self.MARGIN:
            return
        self.renderer.update_scene(data, camera=self.camera)
        rect = mujoco.MjrRect(window.width - self.WIDTH - self.MARGIN, self.MARGIN, self.WIDTH, self.HEIGHT)
        viewer.set_images((rect, self.renderer.render()))

    def close(self):
        self.renderer.close()


class Session:
    def __init__(self, args):
        self.args = args
        self.config = load_config(args.config)
        if getattr(args, "position_scale", None) is not None:
            self.config["mapping"]["position_scale"] = args.position_scale
        self.dt = 1.0 / self.config["control_hz"]
        self.model = scene.build_model()
        self.data = mujoco.MjData(self.model)
        self.output = SimOutput(self.model, self.data, self.dt)
        limits = {s: joint_limits_from_model(self.model, s) for s in SIDES}
        self.core = TeleopCore(self.config, limits, self.dt)
        self.source = make_source(args, self.config)
        self.recorder = None
        if args.record:
            self.recorder = DemonstrationRecorder(
                args.output_dir, {"source": args.source, "control_hz": self.config["control_hz"],
                                  "config": self.config,
                                  "scene": {"base_pos": scene.BASE_POS, "home_arm_q": scene.HOME_ARM_Q,
                                            "object_size_m": scene.OBJECT_SIZE,
                                            "object_mass_kg": scene.OBJECT_MASS}})
        if args.source != "replay":
            for side in SIDES:
                self.core.request_engage(side)
        if args.source == "live":
            self.core.start_imu_fusion()
        self.running = True

    # -- commands ----------------------------------------------------------------
    def handle(self, command):
        core = self.core
        if command == "c":
            if any(l.engaged or l.waiting for l in core.limbs.values()):
                for side in SIDES:
                    core.clutch(side)
                print("both Limbs clutched")
            else:
                for side in SIDES:
                    print(core.toggle(side))
        elif command in ("lc", "rc"):
            print(core.toggle("left" if command == "lc" else "right"))
        elif command == "h":
            core.home()
            print("homing")
        elif command == "r":
            self.reset("discarded")
        elif command in ("ok", "fail"):
            if not self.recorder:
                print("not recording (start with --record)")
                return
            self.reset(command == "ok")
        elif command == "q":
            self.running = False
        elif command:
            print(__doc__.split("Commands")[1])

    def follow_recorded_engagement(self, inputs):
        """Replays reproduce the recording's engaged phases; while clutched (holding
        or going Home) the Limb is put exactly where the recording had it."""
        for side, recorded in inputs.engaged.items():
            limb = self.core.limbs[side]
            if recorded["engaged"] and not limb.engaged:
                self.core.observe(inputs)   # the Neutral pose uses this tick's sample
                self.core.engage(side)
            elif not recorded["engaged"]:
                self.core.clutch(side)
                self.core.hold(side, recorded["arm_q"], recorded["hand_q"])

    def reset(self, outcome):
        """outcome: True / False = save with that success, "discarded" = drop."""
        if self.recorder and self.recorder.active:
            if outcome == "discarded":
                self.recorder.discard()
                print("episode discarded")
            else:
                print(f"saved {self.recorder.save(outcome, 'operator')}")
        self.core.reset()
        self.output.reset()
        print("Scene reset; type c to engage")

    # -- loop --------------------------------------------------------------------
    def tick(self):
        now = time.monotonic()
        inputs = self.source.latest()
        if inputs.engaged is not None:
            self.follow_recorded_engagement(inputs)
        targets, messages = self.core.update(inputs, now)
        for message in messages:
            print(message)
        self.output.send(targets)
        if self.recorder:
            if not self.recorder.active and any(t.engaged for t in targets.values()):
                self.recorder.start()
                print("recording episode")
            self.recorder.add(now, inputs, targets, self.output.measured())
        if isinstance(self.source, ReplaySource) and self.source.finished():
            self.running = False

    def run(self, viewer=None, commands=None, duration=None, inset=None):
        if self.args.source == "replay":
            print("source: replay; following the recorded engage / clutch events")
        else:
            print(f"source: {self.args.source}; each Limb engages once its Wrist board has been "
                  f"visible for {self.config['tracking']['countdown_s']:.0f} s (hold still meanwhile)")
        start = next_tick = time.monotonic()
        try:
            while self.running and (viewer is None or viewer.is_running()):
                while commands is not None and not commands.empty():
                    self.handle(commands.get())
                self.tick()
                if viewer is not None:
                    if inset is not None:
                        inset.draw(viewer, self.data)
                    viewer.sync()
                if duration is not None and time.monotonic() - start >= duration:
                    break
                next_tick += self.dt
                time.sleep(max(0.0, next_tick - time.monotonic()))
        finally:
            if self.recorder and self.recorder.active:
                print(f"saved {self.recorder.save(None, 'quit')}")
            self.core.stop_imu_fusion()
            self.source.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("live", "mock", "replay"), default="live")
    parser.add_argument("--replay", help="Demonstration .npz to replay (with --source replay)")
    parser.add_argument("--loop", action="store_true", help="loop the replay")
    parser.add_argument("--record", action="store_true", help="save Demonstrations")
    parser.add_argument("--output-dir", default=str(TELEOP_DIR / "demonstrations"))
    parser.add_argument("--config", help="teleop config JSON (default: config/teleop.json)")
    parser.add_argument("--position-scale", type=float,
                        help="robot translation per unit of wrist translation (config: mapping.position_scale)")
    parser.add_argument("--headless", action="store_true", help="no viewer, no terminal commands")
    parser.add_argument("--no-inset", action="store_true", help="no horizontal view in the corner of the viewer")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    args = parser.parse_args(argv)

    session = Session(args)
    if args.headless:
        session.run(duration=args.duration)
        return session
    commands = start_command_reader()
    inset = None if args.no_inset else InsetView(session.model)
    try:
        with mujoco.viewer.launch_passive(session.model, session.data) as viewer:
            set_camera(viewer.cam)
            session.run(viewer, commands, args.duration, inset)
    finally:
        if inset is not None:
            inset.close()
    return session


if __name__ == "__main__":
    main()
