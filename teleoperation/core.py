"""The teleoperation core: inputs in, per-Limb joint targets out.

Pure Python, no simulator and no hardware I/O (docs/adr/0002). Each Limb is
engaged (follows its Operator's wrist and Glove), clutched (holds still), or
waiting to engage.

A waiting Limb first drives its Arm back to the Home pose with the Hand open,
then engages once its Wrist board has been visible without interruption for
``tracking.countdown_s``: the Operator's wrist pose at that moment becomes the
Neutral pose, paired with the robot's Home pose. Limbs wait at start-up, after
their Wrist board was lost, and when engaged by command. Clutching by command
holds the Limb still until it is engaged by command again.

When the camera sees a single, ambiguous tag of a Wrist board, the Glove IMU
picks which of the two possible poses is right (teleoperation/imu_fusion.py).
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_dg5fs import scene
from teleoperation.arm_ik import BimanualIK
from teleoperation.finger_retargeting import FingerRetargeter
from teleoperation.imu_fusion import ImuOrientationResolver, load_mount, save_mount
from teleoperation.inputs import SIDES
from teleoperation.wrist_mapping import (RelativeMapper, WristCalibration, WristPoseFilter,
                                         rate_limit)

TELEOP_DIR = Path(__file__).resolve().parent


def load_config(path=None):
    path = Path(path) if path else TELEOP_DIR / "config" / "teleop.json"
    return json.loads(path.read_text())


@dataclass
class LimbTargets:
    arm_q: np.ndarray           # (6,)  scene.ARM_JOINTS order
    hand_q: np.ndarray          # (20,) scene.HAND_JOINTS order
    hand_base: object           # Pose of the Hand base the IK aims for
    engaged: bool
    used_alternative: bool = False   # the Glove IMU picked the estimator's alternative pose


class Limb:
    """Per-Side teleoperation state."""

    def __init__(self, side, config, hand_joint_limits):
        self.side = side
        f = config["filter"]
        self.filter = WristPoseFilter(f["translation_alpha"], f["rotation_alpha"],
                                      f["maximum_jump_m"], f["maximum_jump_deg"])
        calibration = config["wrist_calibration"].get(side)
        self.calibration = (WristCalibration.from_file(TELEOP_DIR / calibration)
                            if calibration else WristCalibration())
        m = config["mapping"]
        self.mapper = RelativeMapper(np.asarray(m["axis_map"]), m["position_scale"])
        self.retarget = FingerRetargeter(side, hand_joint_limits)
        fusion = config.get("imu_fusion", {})
        self.mount_file = (TELEOP_DIR / fusion["mount_file"].format(side=side)) if fusion.get("mount_file") else None
        self.mount_saved = False
        self.resolver = None
        if fusion.get("enabled", True):
            mount = load_mount(self.mount_file) if self.mount_file else None
            self.resolver = ImuOrientationResolver(mount=mount, **fusion.get("resolver", {}))
        self.imu_calibrated_reported = False
        self.used_alternative = False
        self.engaged = False
        self.waiting = False       # going Home, then engaging after the visibility countdown
        self.holding = False       # engaged, Wrist board briefly not seen: holding still
        self.homing = False        # Arm still on its way to the Home pose
        self.visible_since = None
        self.wrist = None          # latest filtered Operator wrist pose
        self.wrist_stamp = None
        self.glove_stamp = None
        self.target = None         # Hand base target
        self.hand_q = np.zeros(20)


class TeleopCore:
    def __init__(self, config, hand_joint_limits, control_dt):
        """hand_joint_limits: {side: {joint name: (lower, upper)}}."""
        self.config = config
        self.dt = control_dt
        ik = config["ik"]
        self.ik = BimanualIK(control_dt, ik["maximum_joint_velocity_rad_s"], ik["posture_cost"],
                             ik["minimum_distance_m"], ik["collision_detection_distance_m"],
                             ik["collision_avoidance"])
        self.limbs = {s: Limb(s, config, hand_joint_limits[s]) for s in SIDES}
        self.imu_max_age = config.get("imu_fusion", {}).get("maximum_imu_age_s", 0.1)
        self.background_fits = False
        self.reset()

    def start_imu_fusion(self):
        """Fit the IMU relation in background threads (live sessions)."""
        self.background_fits = True
        for limb in self.limbs.values():
            if limb.resolver is not None:
                limb.resolver.start()

    def stop_imu_fusion(self):
        for limb in self.limbs.values():
            if limb.resolver is not None:
                limb.resolver.stop()

    # -- state changes ---------------------------------------------------------
    def reset(self):
        """Back to the Home pose, both Limbs clutched."""
        self.ik.reset(scene.home_qpos(self.ik.model))
        for limb in self.limbs.values():
            limb.engaged = limb.waiting = limb.homing = False
            limb.filter.reset()
            limb.wrist = None
            limb.hand_q = np.zeros(20)
            limb.target = self.ik.hand_base_pose(limb.side)

    def request_engage(self, side):
        """Wait to engage: Arm back Home with the Hand open, then engage once the
        Wrist board has been visible for the countdown."""
        limb = self.limbs[side]
        limb.engaged = False
        limb.waiting = True
        limb.homing = True
        limb.visible_since = None

    def engage(self, side):
        """Engage now from wherever the Limb is, capturing the Neutral pose. Used
        once the Limb is Home, and by replays. Returns an error message or None."""
        limb = self.limbs[side]
        if limb.wrist is None or not self._wrist_fresh(limb):
            return f"{side}: no fresh Wrist board pose, cannot engage"
        limb.target = self.ik.hand_base_pose(side)
        limb.mapper.engage(limb.wrist, limb.target)
        limb.engaged = True
        limb.waiting = limb.homing = limb.holding = False
        return None

    def clutch(self, side):
        """Clutch by command: the Limb holds still until engaged by command again."""
        limb = self.limbs[side]
        limb.engaged = limb.waiting = limb.homing = False

    def toggle(self, side):
        """Clutch an engaged or waiting Limb; otherwise ask it to engage. Returns a message."""
        limb = self.limbs[side]
        if limb.engaged or limb.waiting:
            self.clutch(side)
            return f"{side} Limb clutched"
        self.request_engage(side)
        return self._waiting_message(side)

    def home(self):
        """Clutch both Limbs and drive the Arms back to the Home pose, Hands open."""
        for side, limb in self.limbs.items():
            self.clutch(side)
            limb.homing = True

    def hold(self, side, arm_q, hand_q):
        """Put a clutched Limb at the given joints (replays of recorded clutched phases)."""
        self.ik.set_arm_q(side, np.asarray(arm_q, dtype=float))
        limb = self.limbs[side]
        limb.hand_q = np.asarray(hand_q, dtype=float).copy()
        limb.target = self.ik.hand_base_pose(side)

    def _waiting_message(self, side):
        return (f"{side}: going Home; engages once the Wrist board has been visible for "
                f"{self.config['tracking']['countdown_s']:.0f} s (hold still meanwhile)")

    # -- one control tick ------------------------------------------------------
    def _wrist_fresh(self, limb, now=None):
        if limb.wrist_stamp is None:
            return False
        now = limb.wrist_stamp if now is None else now
        return now - limb.wrist_stamp <= self.config["tracking"]["maximum_pose_age_s"]

    def observe(self, inputs):
        """Takes in new Wrist board samples (filter + Wrist calibration). Called by
        update(); call it directly to engage on this tick's sample."""
        for side, limb in self.limbs.items():
            wrist = inputs.wrist[side]
            if (wrist is not None and wrist.stamp != limb.wrist_stamp
                    and wrist.confidence >= self.config["tracking"]["minimum_confidence"]):
                pose = self._resolve(limb, wrist, inputs.imu[side])
                filtered = limb.filter.update(pose)
                if filtered is not None:
                    limb.wrist = limb.calibration.wrist_pose(filtered)
                    limb.wrist_stamp = wrist.stamp

    def _resolve(self, limb, wrist, imu):
        """The camera's pose, or the alternative the Glove IMU says is right."""
        candidates = [wrist.pose] + list(wrist.alternatives)
        limb.used_alternative = False
        if limb.resolver is None or imu is None or abs(wrist.stamp - imu.stamp) > self.imu_max_age:
            return wrist.pose
        rotations = [R.from_matrix(c.rotation) for c in candidates]
        limb.resolver.add(wrist.stamp, rotations, imu.rotation)
        index = limb.resolver.choose(rotations, imu.rotation)
        limb.used_alternative = index > 0
        return candidates[index]

    def update(self, inputs, now):
        """inputs: teleoperation.inputs.Inputs; now: time.monotonic(). Returns
        ({side: LimbTargets}, [messages])."""
        messages = []
        tracking = self.config["tracking"]
        self.observe(inputs)
        for side, limb in self.limbs.items():
            if limb.resolver is not None:
                if not self.background_fits:
                    limb.resolver.maybe_refit(now)
                if limb.resolver.calibrated and not limb.imu_calibrated_reported:
                    limb.imu_calibrated_reported = True
                    messages.append(f"{side}: Glove IMU calibrated against the camera "
                                    f"(residual {limb.resolver.residual_deg:.0f} deg)")
                if limb.resolver.mount_from_fit and limb.mount_file and not limb.mount_saved:
                    save_mount(limb.mount_file, limb.resolver.B, limb.resolver.residual_deg)
                    limb.mount_saved = True
            fresh = self._wrist_fresh(limb, now)
            if limb.engaged:
                messages += self._watch_wrist(side, limb, fresh, now)
            if limb.waiting:
                messages += self._count_down(side, limb, fresh, now)

            glove = inputs.glove[side]
            if limb.engaged and glove is not None and now - glove.stamp <= tracking["maximum_glove_age_s"]:
                limb.hand_q = limb.retarget(glove.ergonomics_deg)

            if limb.engaged:
                s = self.config["safety"]
                limb.target = rate_limit(limb.target, limb.mapper.target(limb.wrist), self.dt,
                                         s["maximum_target_speed_m_s"],
                                         s["maximum_target_rotation_speed_deg_s"])

        not_tracking = {s: self.ik.arm_q(s) for s, l in self.limbs.items() if not l.engaged}
        arm_q = self.ik.step({s: l.target for s, l in self.limbs.items()},
                             {s: l.hand_q for s, l in self.limbs.items()})
        for side, q in not_tracking.items():   # a Limb that is not engaged is only moved Home
            limb = self.limbs[side]
            if limb.homing:
                q = self._step_home(side, q)
            self.ik.set_arm_q(side, q)
            limb.target = self.ik.hand_base_pose(side)
            arm_q[side] = q

        targets = {s: LimbTargets(arm_q[s], l.hand_q.copy(), l.target, l.engaged, l.used_alternative)
                   for s, l in self.limbs.items()}
        return targets, messages

    def _watch_wrist(self, side, limb, fresh, now):
        """An engaged Limb whose Wrist board is not seen holds still (same Neutral
        pose, fingers still follow the Glove) and only after tracking.lost_after_s
        is clutched and sent Home."""
        if fresh:
            if limb.holding:
                limb.holding = False
                return [f"{side}: Wrist board back, Limb continues"]
            return []
        if self._wrist_age(limb, now) > self.config["tracking"].get("lost_after_s", 2.0):
            limb.holding = False
            self.request_engage(side)
            return [f"{side}: Wrist board lost, Limb clutched; " + self._waiting_message(side)]
        if not limb.holding:
            limb.holding = True
            return [f"{side}: Wrist board not seen, Limb holds still "
                    f"(up to {self.config['tracking'].get('lost_after_s', 2.0):.1f} s)"]
        return []

    def _wrist_age(self, limb, now):
        return np.inf if limb.wrist_stamp is None else now - limb.wrist_stamp

    def _count_down(self, side, limb, fresh, now):
        if not fresh:
            lost = limb.visible_since is not None
            limb.visible_since = None
            return [f"{side}: Wrist board lost again, countdown restarts when it is back"] if lost else []
        if limb.visible_since is None:
            limb.visible_since = now
            return [f"{side}: Wrist board visible, engaging in {self.config['tracking']['countdown_s']:.0f} s; "
                    "hold still"]
        if (now - limb.visible_since >= self.config["tracking"]["countdown_s"] and not limb.homing
                and self.engage(side) is None):
            return [f"{side} Limb engaged"]
        return []

    def _step_home(self, side, q):
        """One step of the Arm towards the Home pose; opens the Hand."""
        limb = self.limbs[side]
        limb.hand_q = np.zeros(20)
        max_step = self.config["safety"]["home_joint_speed_rad_s"] * self.dt
        delta = np.asarray(scene.HOME_ARM_Q[side]) - q
        if np.abs(delta).max() <= max_step:
            limb.homing = False
            return np.asarray(scene.HOME_ARM_Q[side], dtype=float)
        return q + np.clip(delta, -max_step, max_step)
