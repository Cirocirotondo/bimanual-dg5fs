#!/usr/bin/env python3
"""Kinematic replay of the Reference retargeted to one Object placement (Viser).

Pick x, y and yaw of the Object in the browser and press Compute: both Arms are
re-solved by IK frame by frame (the same retargeting as the Feasibility map) and
the clip plays back kinematically, with the Object moved along with it. Frames
where a criterion fails are listed and shown: IK residual, joint-limit margin,
joint speed, and Limb-Limb / Limb-Table contacts (with the bodies involved).

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/replay_placement.py \
        [--reference lift_cube_2] [--x 0.05 --y 0 --yaw 10] [--port 8081]
    # then open http://localhost:8081
"""
import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_viser import load_urdf, wxyz  # noqa: E402
from feasibility_map import (LIMIT_MARGIN_RAD, POSITION_TOLERANCE_M, ROTATION_TOLERANCE_RAD,  # noqa: E402
                             SINGULARITY_SIGMA, SPEED_LIMIT_RAD_S)

from bimanual_dg5fs import scene  # noqa: E402
from bimanual_training import REFERENCES_DIR, limbs  # noqa: E402
from bimanual_training.placement import transform_points, transform_quaternions  # noqa: E402
from bimanual_training.retarget import PlacementRetargeter  # noqa: E402

CONTROL_HZ = 50.0


class Replay:
    def __init__(self, args):
        import mujoco
        import viser
        from viser.extras import ViserUrdf

        self.args = args
        self.retargeter = PlacementRetargeter(REFERENCES_DIR / f"{args.reference}.npz", device=args.device)
        self.mujoco = mujoco
        self.model = scene.build_model()
        self.data = mujoco.MjData(self.model)
        scene.reset_to_home(self.model, self.data)
        names = [self.model.body(b).name for b in range(self.model.nbody)]
        self.body_names = names
        self.addresses = [np.array([self.model.joint(f"{side}/{j}").qposadr[0] for j in limbs.JOINTS])
                          for side in limbs.SIDES]
        self.object_adr = self.model.joint("object_free").qposadr[0]
        self.lock = threading.Lock()
        self.clip = None
        self.pending = True
        self.playing = False
        self.last_tick = time.monotonic()

        self.server = viser.ViserServer(port=args.port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid("/grid", width=4.0, height=4.0, cell_size=0.1, position=(0.0, 0.0, -0.36))
        top = scene.TABLE_TOP_Z
        self.server.scene.add_box("/table", dimensions=scene.TABLE_SIZE, color=(150, 120, 90), opacity=0.5,
                                  position=(*scene.TABLE_CENTER_XY, top - scene.TABLE_SIZE[2] / 2.0))
        self.cube = self.server.scene.add_box("/object", dimensions=(scene.OBJECT_SIZE,) * 3, color=(217, 77, 64),
                                              opacity=0.9)
        self.demo_cube = self.server.scene.add_box("/demo_object", dimensions=(scene.OBJECT_SIZE,) * 3,
                                                   color=(60, 180, 90), opacity=0.25)
        self.robots, self.joint_order = {}, {}
        for side in limbs.SIDES:
            position, rotation = limbs.base_pose(side)
            self.server.scene.add_frame(f"/{side}", show_axes=False, position=position, wxyz=wxyz(rotation))
            self.robots[side] = ViserUrdf(self.server, load_urdf(limbs.urdf_path(side)), root_node_name=f"/{side}")
            names = self.robots[side].get_actuated_joint_names()
            self.joint_order[side] = np.array([limbs.JOINTS.index(n) for n in names])
        gui = self.server.gui
        with gui.add_folder("Object placement"):
            self.x = gui.add_slider("x [cm]", min=-25.0, max=25.0, step=0.5, initial_value=args.x * 100)
            self.y = gui.add_slider("y [cm]", min=-25.0, max=25.0, step=0.5, initial_value=args.y * 100)
            self.yaw = gui.add_slider("yaw [deg]", min=-45.0, max=45.0, step=0.5, initial_value=args.yaw)
            self.compute = gui.add_button("Compute")
        with gui.add_folder("Playback"):
            self.frame = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self.play = gui.add_button("Play / pause")
            self.speed = gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
            self.show_demo = gui.add_checkbox("Demo's Object", initial_value=True)
        self.summary = gui.add_markdown("Computing ...")
        self.readout = gui.add_markdown("")
        self.compute.on_click(lambda _: setattr(self, "pending", True))
        # The Object follows the sliders at once (a rigid move); the Arms keep the last
        # computed clip until Compute.
        for slider in (self.x, self.y, self.yaw):
            slider.on_update(lambda _: self.publish(int(self.frame.value)))
        self.frame.on_update(lambda _: self.publish(int(self.frame.value)))
        self.play.on_click(self._toggle)
        self.show_demo.on_update(lambda _: setattr(self.demo_cube, "visible", bool(self.show_demo.value)))

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.0, -2.2, 1.3)
            client.camera.look_at = (0.0, -0.3, 0.2)

    def _toggle(self, _):
        with self.lock:
            self.playing = not self.playing
            self.last_tick = time.monotonic()

    def contacts(self, q, pose):
        """Names of the bodies in Limb-Limb or Limb-Table/floor contact at one frame."""
        m, d = self.model, self.data
        for limb, address in enumerate(self.addresses):
            d.qpos[address] = q[limb]
        x, y, z, qx, qy, qz, qw = pose
        d.qpos[self.object_adr:self.object_adr + 7] = (x, y, z, qw, qx, qy, qz)
        self.mujoco.mj_kinematics(m, d)
        self.mujoco.mj_collision(m, d)
        hits = []
        for c in d.contact[: d.ncon]:
            a = self.body_names[m.geom_bodyid[c.geom1]]
            b = self.body_names[m.geom_bodyid[c.geom2]]
            sa, sb = a.split("/", 1)[0], b.split("/", 1)[0]
            if c.dist >= 0.0 or "object" in (a, b) or sa == sb:
                continue
            if sa in limbs.SIDES or sb in limbs.SIDES:
                hits.append(f"{a} – {b}")
        return sorted(set(hits))

    def run_compute(self):
        x, y, yaw = float(self.x.value) / 100, float(self.y.value) / 100, math.radians(float(self.yaw.value))
        self.summary.content = f"Solving IK for x {x * 100:+.1f} cm, y {y * 100:+.1f} cm, yaw {math.degrees(yaw):+.1f}° ..."
        translation, yaws = torch.tensor([[x, y, 0.0]]), torch.tensor([yaw])
        frames, results = self.retargeter.solve(translation, yaws, frame_stride=1)
        frames = frames.cpu().numpy()
        arm = torch.stack([r["arm_q"][0] for r in results], dim=1).cpu().numpy()                 # (F, 2, 6)
        hand = self.retargeter.hand_q(torch.as_tensor(frames, device=self.retargeter.device)).cpu().numpy()
        q = np.concatenate((arm, hand), axis=-1)
        rd = self.retargeter
        objects = rd.object_track(translation.to(rd.device, rd.dtype), yaws.to(rd.device, rd.dtype),
                                  torch.as_tensor(frames, device=rd.device))[0].cpu().numpy()
        demo = rd.object_pose.cpu().numpy()
        position = torch.stack([r["position_residual"][0] for r in results], 1).amax(1).cpu().numpy()
        rotation = torch.stack([r["rotation_residual"][0] for r in results], 1).amax(1).cpu().numpy()
        margin = torch.stack([torch.minimum(r["arm_q"][0] - r["lower"], r["upper"] - r["arm_q"][0]).amin(-1)
                              for r in results], 1).amin(1).cpu().numpy()
        speed = np.r_[0.0, (np.abs(np.diff(arm, axis=0)) * CONTROL_HZ).max(axis=(1, 2))]
        sigma = torch.stack([r["sigma_min"][0] for r in results], 1).cpu().numpy()             # (F, 2)
        contacts = [self.contacts(q[k], objects[k]) for k in range(len(frames))]
        problems = {
            "IK": np.flatnonzero((position > POSITION_TOLERANCE_M) | (rotation > ROTATION_TOLERANCE_RAD)),
            "joint limit": np.flatnonzero(margin < LIMIT_MARGIN_RAD),
            "joint speed": np.flatnonzero(speed > SPEED_LIMIT_RAD_S),
            "singularity": np.flatnonzero(sigma.min(axis=1) < SINGULARITY_SIGMA),
            "collision": np.flatnonzero([bool(c) for c in contacts]),
        }
        with self.lock:
            self.clip = dict(q=q, objects=objects, demo=demo, position=position, rotation=rotation, margin=margin, sigma=sigma,
                             speed=speed, contacts=contacts, problems=problems, placement=(x, y, yaw))
            self.playing = False
        self.frame.max = len(frames) - 1
        self.frame.value = 0
        rows = "\n".join(f"| {name} | {len(idx)} | {'–' if not len(idx) else f'{idx[0]}…{idx[-1]}'} |"
                         for name, idx in problems.items())
        verdict = "**FEASIBLE**" if not any(len(i) for i in problems.values()) else "**not feasible**"
        self.summary.content = (
            f"{verdict}: x {x * 100:+.1f} cm, y {y * 100:+.1f} cm, yaw {math.degrees(yaw):+.1f}°\n\n"
            f"| criterion | frames failing | range |\n|---|---|---|\n{rows}\n\n"
            f"Worst: IK {position.max() * 1000:.1f} mm / {rotation.max() * 1000:.1f} mrad, "
            f"margin {margin.min():.3f} rad, speed {speed.max():.2f} rad/s, "
            f"sigma_min {sigma.min():.3f} (threshold {SINGULARITY_SIGMA})")
        self.publish(0)

    def placement(self):
        return float(self.x.value) / 100, float(self.y.value) / 100, math.radians(float(self.yaw.value))

    def placed_object(self, frame):
        """The Object at ``frame`` moved by the sliders' placement (x y z + xyzw)."""
        x, y, yaw = self.placement()
        rd = self.retargeter
        pose = rd.object_pose[frame]
        translation = torch.tensor([x, y, 0.0], dtype=rd.dtype, device=rd.device)
        yaw = torch.tensor(yaw, dtype=rd.dtype, device=rd.device)
        position = transform_points(pose[:3], translation, yaw, rd.pivot)
        quat = transform_quaternions(pose[3:7], yaw)
        return torch.cat((position, quat)).cpu().numpy()

    def publish(self, index):
        with self.lock:
            clip = self.clip
        if clip is None:
            frame = 0
            x, y, z, qx, qy, qz, qw = self.placed_object(frame)
            self.cube.position, self.cube.wxyz = (x, y, z), (qw, qx, qy, qz)
            return
        k = min(index, len(clip["q"]) - 1)
        stale = any(abs(a - b) > 1e-9 for a, b in zip(self.placement(), clip["placement"]))
        for limb, side in enumerate(limbs.SIDES):
            self.robots[side].update_cfg(clip["q"][k, limb][self.joint_order[side]])
        x, y, z, qx, qy, qz, qw = self.placed_object(k) if stale else clip["objects"][k]
        self.cube.position, self.cube.wxyz = (x, y, z), (qw, qx, qy, qz)
        x, y, z, qx, qy, qz, qw = clip["demo"][k]
        self.demo_cube.position, self.demo_cube.wxyz = (x, y, z), (qw, qx, qy, qz)
        hits = clip["contacts"][k]
        self.cube.color = (220, 40, 40) if hits else (217, 77, 64)
        self.readout.content = (
            ("**The Arms show the previous placement: press Compute.**  \n" if stale else "")
            + f"**Frame {k}/{len(clip['q']) - 1}**  \n"
            f"IK {clip['position'][k] * 1000:.2f} mm / {clip['rotation'][k] * 1000:.1f} mrad · "
            f"margin {clip['margin'][k]:.3f} rad · speed {clip['speed'][k]:.2f} rad/s · "
            f"sigma_min {' / '.join(f'{v:.3f}' for v in clip['sigma'][k])} ({' / '.join(limbs.SIDES)})  \n"
            + ("**Contacts:** " + "; ".join(hits) if hits else "no contacts"))

    def loop(self):
        print(f"Open http://localhost:{self.args.port}", flush=True)
        while True:
            if self.pending:
                self.pending = False
                self.run_compute()
                continue
            time.sleep(1.0 / 120.0)
            with self.lock:
                if not self.playing or self.clip is None:
                    continue
                now = time.monotonic()
                advance = int((now - self.last_tick) * CONTROL_HZ * float(self.speed.value))
                if advance <= 0:
                    continue
                self.last_tick = now
                following = (int(self.frame.value) + advance) % len(self.clip["q"])
            self.frame.value = following


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default="lift_cube_2")
    parser.add_argument("--x", type=float, default=0.0, help="[m]")
    parser.add_argument("--y", type=float, default=0.0, help="[m]")
    parser.add_argument("--yaw", type=float, default=0.0, help="[deg]")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--device", default="cuda")
    Replay(parser.parse_args()).loop()


if __name__ == "__main__":
    main()
