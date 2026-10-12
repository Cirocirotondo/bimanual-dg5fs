"""Outputs: where per-Limb joint targets go (docs/adr/0002).

An output takes ``{side: core.LimbTargets}`` every control tick and reports
what the robot actually did. Only the simulation exists today; a real-robot
output (Hand targets to Tesollo's ROS 2 driver, Arm targets to the UR
impedance controller) implements the same three methods.
"""
import mujoco
import numpy as np

from bimanual_dg5fs import scene


class SimOutput:
    """Drives the Scene's position actuators and steps the physics."""

    def __init__(self, model, data, control_dt):
        self.model = model
        self.data = data
        self.substeps = max(1, int(round(control_dt / model.opt.timestep)))
        self.actuator = {s: {"arm": [model.actuator(f"{s}/{j}").id for j in scene.ARM_JOINTS],
                             "hand": [model.actuator(f"{s}/{j}").id for j in scene.HAND_JOINTS]}
                         for s in scene.SIDES}
        self.qadr = {s: {"arm": [model.joint(f"{s}/{j}").qposadr[0] for j in scene.ARM_JOINTS],
                         "hand": [model.joint(f"{s}/{j}").qposadr[0] for j in scene.HAND_JOINTS]}
                     for s in scene.SIDES}
        self.object_qadr = model.joint("object_free").qposadr[0]
        self.reset()

    def reset(self):
        scene.reset_to_home(self.model, self.data)

    def send(self, targets):
        for side, t in targets.items():
            self.data.ctrl[self.actuator[side]["arm"]] = t.arm_q
            self.data.ctrl[self.actuator[side]["hand"]] = t.hand_q
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def measured(self):
        """{side: {"arm_q", "hand_q", "hand_base" (x y z qx qy qz qw)}, "object": (7,)}"""
        out = {}
        for side in scene.SIDES:
            body = self.data.body(f"{side}/link_base")
            out[side] = {
                "arm_q": self.data.qpos[self.qadr[side]["arm"]].copy(),
                "hand_q": self.data.qpos[self.qadr[side]["hand"]].copy(),
                "hand_base": np.r_[body.xpos, body.xquat[1:], body.xquat[0]],
            }
        q = self.data.qpos[self.object_qadr:self.object_qadr + 7]
        out["object"] = np.r_[q[:3], q[4:7], q[3]]
        return out
