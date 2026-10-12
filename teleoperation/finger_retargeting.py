"""Finger retargeting: MANUS ergonomics angles -> DG-5F-S Hand joint targets.

A port of TESOLLO's official per-joint formula
(tesollo_ros2/dg5f_ros2/dg5f_driver/script/manus_retarget/manus_retarget.py,
``_compute_mqd_from_q``), which the UR5 PC's DG-5F-S adapter
(dg5f_s_manus_retarget.py) uses unchanged for the right hand. The left hand
uses the official left-hand directions and backward-bending rules. The result
is clamped to the Hand's joint limits (the Limb URDF's).
"""
import numpy as np

# MANUS ErgonomicsDataType order, per Glove (degrees).
ERGONOMICS = (
    "ThumbMCPSpread", "ThumbMCPStretch", "ThumbPIPStretch", "ThumbDIPStretch",
    "IndexSpread", "IndexMCPStretch", "IndexPIPStretch", "IndexDIPStretch",
    "MiddleSpread", "MiddleMCPStretch", "MiddlePIPStretch", "MiddleDIPStretch",
    "RingSpread", "RingMCPStretch", "RingPIPStretch", "RingDIPStretch",
    "PinkySpread", "PinkyMCPStretch", "PinkyPIPStretch", "PinkyDIPStretch",
)

# Hand joints in output order: finger 1 (thumb) .. 5, joint 1 .. 4.
HAND_JOINTS = tuple(f"joint_{f}_{j}" for f in range(1, 6) for j in range(1, 5))

CALIBRATION = (1.0, 1.6, 1.3, 1.3,
               1.0, 1.0, 1.3, 1.7,
               1.0, 1.0, 1.3, 1.7,
               1.0, 1.0, 1.3, 1.7,
               1.0, 1.0, 1.0, 1.0)

DIRECTION = {
    "right": (1, -1, 1, 1,
              -1, 1, 1, 1,
              -1, 1, 1, 1,
              -1, 1, 1, 1,
              1, -1, 1, 1),
    "left": (-1, 1, -1, -1,
             1, 1, 1, 1,
             1, 1, 1, 1,
             1, 1, 1, 1,
             -1, 1, 1, 1),
}

# Joints that may only bend one way (TESOLLO's backward-bending prevention):
# these must stay <= 0; every other joint not in NO_ONE_SIDED_CLAMP must stay >= 0.
NON_POSITIVE = {"right": {1}, "left": {0, 2, 3}}
NO_ONE_SIDED_CLAMP = {4, 8, 12, 16, 17}


def tesollo_targets(side, q_deg):
    """TESOLLO's formula for one Glove: 20 ergonomics angles [deg] -> 20 joint
    targets [rad] in HAND_JOINTS order, before clamping to joint limits."""
    q = np.zeros(20)
    q[: min(20, len(q_deg))] = np.asarray(q_deg, dtype=float)[:20]

    qd = np.zeros(20)
    qd[0] = 58.5 - q[1]
    qd[1] = q[0] + 20.0
    qd[2] = q[2]
    qd[3] = 0.5 * (q[2] + q[3])
    qd[4], qd[5], qd[6], qd[7] = q[4], q[5], q[6] - 40.0, q[7]
    qd[8], qd[9], qd[10], qd[11] = q[8], q[9], q[10] - 30.0, q[11]
    qd[12], qd[13], qd[14], qd[15] = q[12], q[13], q[14], q[15]
    # Pinky: joint 5_1 folds the palm, driven by the spread magnitude.
    qd[16] = abs(q[16]) * 2.0 if (q[17] > 55.0 and q[18] > 25.0) else abs(q[16]) / 1.5
    qd[17], qd[18], qd[19] = q[16], q[17], q[18]

    out = np.radians(qd) * np.asarray(CALIBRATION) * np.asarray(DIRECTION[side])
    for i in range(20):
        if i in NO_ONE_SIDED_CLAMP:
            continue
        out[i] = min(out[i], 0.0) if i in NON_POSITIVE[side] else max(out[i], 0.0)
    return out


class FingerRetargeter:
    """Finger retargeting for one Side, clamped to that Hand's joint limits."""

    def __init__(self, side, joint_limits):
        """joint_limits: {joint name (unprefixed): (lower, upper)} in radians."""
        self.side = side
        self.lower = np.array([joint_limits[j][0] for j in HAND_JOINTS])
        self.upper = np.array([joint_limits[j][1] for j in HAND_JOINTS])

    def __call__(self, ergonomics_deg):
        return np.clip(tesollo_targets(self.side, ergonomics_deg), self.lower, self.upper)


def joint_limits_from_model(model, side):
    """{joint name: (lower, upper)} for one Hand of the Scene model."""
    return {j: tuple(model.joint(f"{side}/{j}").range) for j in HAND_JOINTS}

