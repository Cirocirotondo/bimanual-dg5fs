#!/usr/bin/env python3
"""Interactive evaluation of a Motion imitation policy in the browser (Viser).

Isaac Lab runs headless in this process; Viser draws what it measures. In the
browser you pick a run and a checkpoint (or the feed-forward replay of the
Reference), the start frame, and press Run: one deterministic rollout (mean
actions) is streamed live and then kept for frame-by-frame scrubbing.

The Limbs are drawn with their meshes, the Reference as a transparent green
ghost of both Limbs (``--ghost-offset`` puts it beside them instead of on top).
Hand-base frames show where each palm is and where the Reference wants it.
The table and the Reference's cube are context only: Training stage 1 has no
Object and no table to collide with.

The numbers are the ones scripts/evaluate.py prints (worst Limb at each frame),
plus per-Limb curves of the Hand-base position error and the hand joint RMS
error.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/evaluate_viser.py [--port 8080]
    # then open http://localhost:8080
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from bimanual_dg5fs import scene
from bimanual_training import ASSETS_DIR, LOGS_DIR, limbs
from bimanual_training.launch import add_env_arguments, make_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import HAND_SUCCESS_RAD, PALM_SUCCESS_M, load_policy  # noqa: E402

REPLAY = "(feed-forward replay of the Reference)"
RUNS_DIR = LOGS_DIR / "motion_imitation"
CONTROL_HZ = 50.0
GHOST_RGBA = (0.2, 0.85, 0.35, 0.35)
LIMB_COLORS = {"left": "#d9480f", "right": "#1971c2"}


def wxyz(xyzw):
    x, y, z, w = (float(v) for v in xyzw)
    return (w, x, y, z)


def load_urdf(path: Path):
    """The Limb URDF with mesh paths resolved against assets/."""
    import yourdfpy

    def resolve(fname: str) -> str:
        candidate = Path(fname)
        if candidate.is_absolute():
            return str(candidate)
        for base in (ASSETS_DIR, path.parent):
            if (base / fname).is_file():
                return str(base / fname)
        return fname

    return yourdfpy.URDF.load(str(path), filename_handler=resolve)


def list_runs():
    """Run directories with at least one checkpoint, newest first."""
    runs = [d for d in RUNS_DIR.glob("*") if d.is_dir() and any(d.glob("model_*.pt"))]
    return sorted((d.name for d in runs), reverse=True)


def list_checkpoints(run):
    models = (RUNS_DIR / run).glob("model_*.pt")
    return sorted((m.name for m in models), key=lambda n: int(n[6:-3]), reverse=True)


@dataclass
class Frame:
    reference_index: int
    q: np.ndarray                     # (2, 26) Limb order
    reference_q: np.ndarray           # (2, 26)
    palm_position: np.ndarray         # (2, 3)
    palm_quat: np.ndarray             # (2, 4) xyzw
    palm_error_m: np.ndarray          # (2,) Hand-base position error
    hand_rms_rad: np.ndarray          # (2,)
    fingertip_error_m: np.ndarray     # (2,) mean over the five fingertips


@dataclass
class Rollout:
    label: str
    start: int
    frames: list = field(default_factory=list)
    termination: int | None = None    # Reference frame of the early termination
    finished: bool = False


class Evaluator:
    def __init__(self, args):
        import viser
        from viser.extras import ViserUrdf

        args.num_envs = 1
        self.args = args
        self.env, self.stack = make_env(args, rsi_start_index=0)
        self.env.reset()
        self.reference = self.env.reference
        self.reference_np = {
            "q": self.reference.q.cpu().numpy(),
            "palm_position": self.reference.palm_position.cpu().numpy(),
            "palm_quat": self.reference.palm_quat.cpu().numpy(),
            "object_pose": self.reference.object_pose.cpu().numpy(),
        }
        self.thresholds = (self.env.cfg.palm_keypoint_threshold_m, self.env.cfg.hand_position_threshold_rad)
        self.policies = {}
        self.lock = threading.Lock()
        self.rollout: Rollout | None = None
        self.pending: tuple | None = None
        self.stop_requested = False
        self.playing = False
        self.last_tick = time.monotonic()

        self.server = viser.ViserServer(port=args.port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid("/grid", width=4.0, height=4.0, cell_size=0.1, position=(0.0, 0.0, -0.36))
        top = scene.TABLE_TOP_Z
        self.server.scene.add_box("/table", dimensions=scene.TABLE_SIZE, color=(150, 120, 90), opacity=0.5,
                                  position=(*scene.TABLE_CENTER_XY, top - scene.TABLE_SIZE[2] / 2.0))
        self.reference_cube = self.server.scene.add_box(
            "/reference_cube", dimensions=(scene.OBJECT_SIZE,) * 3, color=(60, 180, 90), opacity=0.35)
        ghost_shift = np.array([0.0, float(args.ghost_offset), 0.0])
        self.robots, self.ghosts, self.joint_order = {}, {}, {}
        self.palm_frames, self.reference_palm_frames = {}, {}
        for side in limbs.SIDES:
            position, rotation = limbs.base_pose(side)
            urdf = load_urdf(limbs.urdf_path(side))
            self.server.scene.add_frame(f"/{side}", show_axes=False, position=position, wxyz=wxyz(rotation))
            self.server.scene.add_frame(f"/ghost_{side}", show_axes=False,
                                        position=np.asarray(position) + ghost_shift, wxyz=wxyz(rotation))
            self.robots[side] = ViserUrdf(self.server, urdf, root_node_name=f"/{side}")
            self.ghosts[side] = ViserUrdf(self.server, urdf, root_node_name=f"/ghost_{side}",
                                          mesh_color_override=GHOST_RGBA)
            names = self.robots[side].get_actuated_joint_names()
            self.joint_order[side] = np.array([limbs.JOINTS.index(n) for n in names])
            self.palm_frames[side] = self.server.scene.add_frame(
                f"/palm_{side}", axes_length=0.08, axes_radius=0.005)
            self.reference_palm_frames[side] = self.server.scene.add_frame(
                f"/reference_palm_{side}", axes_length=0.08, axes_radius=0.002, origin_color=(40, 200, 80))
        self.ghost_shift = ghost_shift
        self._build_gui()

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.0, -2.2, 1.3)
            client.camera.look_at = (0.0, -0.3, 0.2)

        self._show_reset_pose(0)

    # ------------------------------------------------------------------ GUI

    def _build_gui(self):
        gui = self.server.gui
        with gui.add_folder("Rollout"):
            runs = list_runs()
            self.run_dropdown = gui.add_dropdown("Run", options=[REPLAY] + runs,
                                                 initial_value=runs[0] if runs else REPLAY)
            self.checkpoint_dropdown = gui.add_dropdown("Checkpoint", options=["-"])
            self.start_slider = gui.add_slider("Start frame", min=0, max=self.reference.last_index - 1,
                                               step=1, initial_value=0)
            self.terminate = gui.add_checkbox("Terminate as in training", initial_value=True,
                                              hint="Off: the policy runs to the end of the Reference whatever its error")
            self.run_button = gui.add_button("Run")
            self.stop_button = gui.add_button("Stop", disabled=True)
        with gui.add_folder("Playback"):
            self.frame_slider = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self.play_button = gui.add_button("Play / pause")
            self.speed = gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
            self.loop = gui.add_checkbox("Loop", initial_value=True)
            self.show_ghost = gui.add_checkbox("Reference ghost", initial_value=True)
            self.show_frames = gui.add_checkbox("Hand-base frames", initial_value=True)
        self.summary = gui.add_markdown("Pick a run and press **Run**.")
        self.readout = gui.add_markdown("")
        self.palm_plot = None
        self.hand_plot = None

        self.run_dropdown.on_update(self._update_checkpoints)
        self.run_button.on_click(self._request_run)
        self.stop_button.on_click(self._request_stop)
        self.frame_slider.on_update(lambda _: self._publish(int(self.frame_slider.value)))
        self.play_button.on_click(self._toggle_play)
        self.show_ghost.on_update(self._toggle_visibility)
        self.show_frames.on_update(self._toggle_visibility)
        self._update_checkpoints(None)

    def _update_checkpoints(self, _):
        run = self.run_dropdown.value
        options = ["-"] if run == REPLAY else list_checkpoints(run)
        self.checkpoint_dropdown.options = options
        self.checkpoint_dropdown.value = options[0]

    def _toggle_visibility(self, _):
        for side in limbs.SIDES:
            self.ghosts[side].show_visual = bool(self.show_ghost.value)
            self.palm_frames[side].visible = bool(self.show_frames.value)
            self.reference_palm_frames[side].visible = bool(self.show_frames.value)
        self.reference_cube.visible = bool(self.show_ghost.value)

    def _toggle_play(self, _):
        with self.lock:
            self.playing = not self.playing
            self.last_tick = time.monotonic()

    def _request_stop(self, _):
        self.stop_requested = True

    def _request_run(self, _):
        run = self.run_dropdown.value
        checkpoint = None if run == REPLAY else RUNS_DIR / run / self.checkpoint_dropdown.value
        label = "replay" if checkpoint is None else f"{run}/{checkpoint.name}"
        with self.lock:
            self.pending = (checkpoint, int(self.start_slider.value), bool(self.terminate.value), label)

    # -------------------------------------------------------------- rollout

    def _policy(self, checkpoint):
        if checkpoint not in self.policies:
            self.summary.content = f"Loading `{checkpoint.name}` ..."
            self.policies[checkpoint] = load_policy(self.env, str(checkpoint), self.args.device)
        return self.policies[checkpoint]

    def _capture(self):
        """The current state and its errors against the Reference frame the episode is at."""
        env = self.env
        index = int(env.reference_index[0])
        palm_position, palm_quat = env.palm_pose()
        q = env.q[0]
        reference_q = self.reference.q[index]
        hand_error = q[:, limbs.NUM_ARM:] - reference_q[:, limbs.NUM_ARM:]
        return Frame(
            reference_index=index,
            q=q.cpu().numpy(),
            reference_q=reference_q.cpu().numpy(),
            palm_position=palm_position[0].cpu().numpy(),
            palm_quat=palm_quat[0].cpu().numpy(),
            palm_error_m=(palm_position[0] - self.reference.palm_position[index]).norm(dim=-1).cpu().numpy(),
            hand_rms_rad=hand_error.square().mean(dim=-1).sqrt().cpu().numpy(),
            fingertip_error_m=(env.fingertips()[0] - self.reference.fingertips[index]).norm(dim=-1).mean(-1)
            .cpu().numpy(),
        )

    def _rollout(self, checkpoint, start, terminate, label):
        env = self.env
        env.cfg.rsi_start_index = start
        if terminate:
            env.cfg.palm_keypoint_threshold_m, env.cfg.hand_position_threshold_rad = self.thresholds
        else:
            env.cfg.palm_keypoint_threshold_m = env.cfg.hand_position_threshold_rad = float("inf")
        try:
            policy = None if checkpoint is None else self._policy(checkpoint)
        except Exception as error:  # a checkpoint of another observation layout, a broken file
            self.summary.content = f"**Cannot load** `{checkpoint}`: {error}"
            return
        rollout = Rollout(label=label, start=start)
        with self.lock:
            self.rollout, self.playing = rollout, False
        self.run_button.disabled, self.stop_button.disabled = True, False
        self.stop_requested = False
        observations, _ = env.reset()
        from tensordict import TensorDict

        # no_grad, not inference_mode: inference tensors left in the environment's
        # buffers would stop the next checkpoint's runner from initialising.
        with torch.no_grad():
            while not self.stop_requested:
                frame = self._capture()
                with self.lock:
                    rollout.frames.append(frame)
                self._publish(len(rollout.frames) - 1, live=True)
                if policy is None:
                    actions = env.reference_actions()
                else:
                    actions = policy(TensorDict(observations, batch_size=[env.num_envs]))
                observations, _, early, timeout, _ = env.step(actions)
                if bool(early[0]):
                    rollout.termination = frame.reference_index + 1
                    break
                if bool(timeout[0]):
                    break
        rollout.finished = True
        env.cfg.palm_keypoint_threshold_m, env.cfg.hand_position_threshold_rad = self.thresholds
        self.run_button.disabled, self.stop_button.disabled = False, True
        self.frame_slider.max = max(1, len(rollout.frames) - 1)
        self.frame_slider.value = 0
        self._summarise(rollout)
        self._plot(rollout)

    # ------------------------------------------------------------- display

    def _show_reset_pose(self, index):
        q = self.reference_np["q"][index]
        for limb, side in enumerate(limbs.SIDES):
            self.robots[side].update_cfg(q[limb][self.joint_order[side]])
            self.ghosts[side].update_cfg(q[limb][self.joint_order[side]])
        self._place_reference(index)

    def _place_reference(self, index):
        for limb, side in enumerate(limbs.SIDES):
            frame = self.reference_palm_frames[side]
            frame.position = self.reference_np["palm_position"][index, limb]
            frame.wxyz = wxyz(self.reference_np["palm_quat"][index, limb])
        x, y, z, qx, qy, qz, qw = self.reference_np["object_pose"][index]
        self.reference_cube.position = np.array([x, y, z]) + self.ghost_shift
        self.reference_cube.wxyz = (qw, qx, qy, qz)

    def _publish(self, index, live=False):
        with self.lock:
            rollout = self.rollout
            if rollout is None or not rollout.frames:
                return
            frame = rollout.frames[min(index, len(rollout.frames) - 1)]
            total = len(rollout.frames)
        for limb, side in enumerate(limbs.SIDES):
            self.robots[side].update_cfg(frame.q[limb][self.joint_order[side]])
            self.ghosts[side].update_cfg(frame.reference_q[limb][self.joint_order[side]])
            self.palm_frames[side].position = frame.palm_position[limb]
            self.palm_frames[side].wxyz = wxyz(frame.palm_quat[limb])
        self._place_reference(frame.reference_index)
        status = "running" if live else f"step {index}/{total - 1}"
        rows = "\n".join(
            f"| {side} | {100 * frame.palm_error_m[limb]:.2f} | {frame.hand_rms_rad[limb]:.3f} | "
            f"{100 * frame.fingertip_error_m[limb]:.2f} |" for limb, side in enumerate(limbs.SIDES))
        self.readout.content = (
            f"**{status}**, Reference frame {frame.reference_index}/{self.reference.last_index}\n\n"
            "| Limb | palm [cm] | hand RMS [rad] | tips [cm] |\n|---|---|---|---|\n" + rows)

    def _summarise(self, rollout):
        frames = rollout.frames
        palm = np.array([f.palm_error_m.max() for f in frames])
        hand = np.array([f.hand_rms_rad.max() for f in frames])
        tips = np.array([f.fingertip_error_m.max() for f in frames])
        span = self.reference.last_index - rollout.start
        end = (f"**early termination at frame {rollout.termination}**" if rollout.termination is not None
               else "stopped" if len(frames) < span else "reached the end")
        success = (rollout.termination is None and len(frames) >= span
                   and palm.mean() < PALM_SUCCESS_M and hand.mean() < HAND_SUCCESS_RAD)
        self.summary.content = (
            f"**{rollout.label}** from frame {rollout.start}: {end} ({len(frames)}/{span} frames)\n\n"
            f"| worst Limb | mean | max |\n|---|---|---|\n"
            f"| palm position [cm] | {100 * palm.mean():.2f} | {100 * palm.max():.2f} |\n"
            f"| hand joint RMS [rad] | {hand.mean():.3f} | {hand.max():.3f} |\n"
            f"| fingertips [cm] | {100 * tips.mean():.2f} | {100 * tips.max():.2f} |\n\n"
            + ("**SUCCESS**" if success else
               f"not yet (success: whole Reference, palm < {100 * PALM_SUCCESS_M:.0f} cm, "
               f"hand < {HAND_SUCCESS_RAD} rad)"))

    def _plot(self, rollout):
        frames = rollout.frames
        x = np.array([f.reference_index for f in frames], dtype=np.float64)
        palm = np.array([f.palm_error_m for f in frames]) * 100.0
        hand = np.array([f.hand_rms_rad for f in frames])
        series = ({"label": "Reference frame"},) + tuple(
            {"label": side, "stroke": LIMB_COLORS[side], "width": 2} for side in limbs.SIDES)
        for handle in (self.palm_plot, self.hand_plot):
            if handle is not None:
                handle.remove()
        self.palm_plot = self.server.gui.add_uplot(
            (x, palm[:, 0], palm[:, 1]), series, title="Hand-base position error [cm]", aspect=1.8)
        self.hand_plot = self.server.gui.add_uplot(
            (x, hand[:, 0], hand[:, 1]), series, title="Hand joint RMS error [rad]", aspect=1.8)

    # ----------------------------------------------------------------- loop

    def run(self):
        print(f"Open http://localhost:{self.args.port}", flush=True)
        try:
            while True:
                with self.lock:
                    pending, self.pending = self.pending, None
                if pending is not None:
                    self._rollout(*pending)
                    continue
                time.sleep(1.0 / 120.0)
                with self.lock:
                    rollout = self.rollout
                    if not self.playing or rollout is None or not rollout.finished or not rollout.frames:
                        continue
                    now = time.monotonic()
                    advance = int((now - self.last_tick) * CONTROL_HZ * float(self.speed.value))
                    if advance <= 0:
                        continue
                    self.last_tick = now
                    total = len(rollout.frames)
                    following = int(self.frame_slider.value) + advance
                    if following >= total:
                        if self.loop.value:
                            following %= total
                        else:
                            following, self.playing = total - 1, False
                self.frame_slider.value = following
        except KeyboardInterrupt:
            print("Evaluator stopped.")
        finally:
            self.env.close()
            self.stack.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_arguments(parser)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ghost-offset", type=float, default=0.0,
                        help="draw the Reference ghost this far along y [m] instead of on the Limbs")
    Evaluator(parser.parse_args()).run()


if __name__ == "__main__":
    main()
