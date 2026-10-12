"""Input sources: Glove finger angles and Wrist board poses, per Side.

Every source answers ``latest()`` with an :class:`Inputs` snapshot holding the
newest sample per Side (or None). Stamps are ``time.monotonic()`` seconds.

* :class:`LiveSource` - the MANUS client (Nando's SDKClient_Linux, ZMQ PUSH on
  tcp 8001: 40 comma-separated ergonomics angles in degrees, left glove first;
  tcp 8000: per Glove ``id,`` + 25 skeleton nodes x (px py pz qx qy qz qw), node 0
  carrying the Glove IMU's hand orientation) and the tag pose estimator (ZMQ PUB
  on tcp 5557: JSON ``{"timestamp", "poses": {board_index: {"position",
  "rotation_matrix", "confidence", "alternatives"}}}`` in the World board frame;
  ``alternatives`` are the other poses of an ambiguous single tag).
* :class:`MockSource` - scripted wrist motion and finger curl, no hardware.
* :class:`ReplaySource` - the raw inputs stored in a Demonstration.
"""
import json
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R

from teleoperation.wrist_mapping import Pose

SIDES = ("right", "left")


@dataclass
class WristSample:
    pose: Pose                  # Wrist board pose in the World board frame
    confidence: float
    stamp: float
    alternatives: list = field(default_factory=list)   # other Poses explaining the image as well


@dataclass
class ImuSample:
    rotation: R                 # Glove IMU hand orientation (its own reference frame)
    stamp: float


@dataclass
class GloveSample:
    ergonomics_deg: np.ndarray  # (20,), MANUS ErgonomicsDataType order
    stamp: float


@dataclass
class Inputs:
    wrist: dict = field(default_factory=lambda: {s: None for s in SIDES})
    glove: dict = field(default_factory=lambda: {s: None for s in SIDES})
    imu: dict = field(default_factory=lambda: {s: None for s in SIDES})
    # Only replays set this, per Side: {"engaged", "arm_q", "hand_q"} as recorded.
    # The session reproduces it instead of running its own countdown and commands.
    engaged: dict = None


# -----------------------------------------------------------------------------
# Live hardware
# -----------------------------------------------------------------------------
def parse_ergonomics(message):
    """40 floats (left 0..19, right 20..39) -> {side: (20,) or None}. A Glove that
    is not connected is sent as all zeros and reported as None."""
    values = np.array([float(v) for v in message.split(",")])
    if values.size != 40:
        raise ValueError(f"expected 40 ergonomics values, got {values.size}")
    out = {}
    for side, block in (("left", values[:20]), ("right", values[20:])):
        out[side] = None if np.all(np.abs(block) < 1e-6) else block
    return out


def _pose(entry):
    if "rotation_matrix" in entry:
        rotation = np.asarray(entry["rotation_matrix"], dtype=float)
    elif "quaternion_wxyz" in entry:
        rotation = R.from_quat(entry["quaternion_wxyz"], scalar_first=True).as_matrix()
    else:
        return None
    return Pose(np.asarray(entry["position"], dtype=float), rotation)


def parse_board_poses(message):
    """Pose estimator JSON -> {board index (str): (Pose, confidence, [alternative Poses])}."""
    poses = {}
    for key, entry in message.get("poses", {}).items():
        pose = _pose(entry)
        if pose is None:
            continue
        alternatives = [p for p in (_pose(a) for a in entry.get("alternatives", [])) if p is not None]
        poses[str(key)] = (pose, float(entry.get("confidence", 1.0)), alternatives)
    return poses


SKELETON_NODES = 25


def parse_skeleton(message):
    """MANUS skeleton line -> {glove id (lower-case hex): IMU orientation (Rotation)}."""
    fields = message.split(",")
    out, i, block = {}, 0, 1 + SKELETON_NODES * 7
    while i + block <= len(fields):
        wrist = np.array(fields[i + 1:i + 8], dtype=float)
        out[fields[i].strip().lower()] = R.from_quat(wrist[3:7])
        i += block
    return out


class _ZmqReceiver(threading.Thread):
    def __init__(self, address, socket_type, handle):
        super().__init__(daemon=True)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(socket_type)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        if socket_type == zmq.SUB:
            self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(address)
        self._handle = handle
        self._running = True

    def run(self):
        while self._running:
            try:
                message = self._socket.recv()
            except zmq.Again:
                continue
            try:
                self._handle(message)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                print(f"[inputs] dropped malformed message: {error}")
        self._socket.close(linger=0)

    def stop(self):
        self._running = False


class LiveSource:
    def __init__(self, glove_address, pose_address, wrist_board_index, skeleton_address=None, glove_ids=None):
        """wrist_board_index: {side: board index in the pose estimator config};
        glove_ids: {side: MANUS glove id} for the IMU orientations on skeleton_address."""
        self._lock = threading.Lock()
        self._inputs = Inputs()
        self._board_to_side = {str(i): side for side, i in wrist_board_index.items()}
        self._glove_to_side = {g.lower(): side for side, g in (glove_ids or {}).items()}
        self._receivers = [
            _ZmqReceiver(glove_address, zmq.PULL, self._on_glove),
            _ZmqReceiver(pose_address, zmq.SUB, self._on_poses),
        ]
        if skeleton_address and glove_ids:
            self._receivers.append(_ZmqReceiver(skeleton_address, zmq.PULL, self._on_skeleton))
        for receiver in self._receivers:
            receiver.start()

    def _on_glove(self, message):
        now = time.monotonic()
        parsed = parse_ergonomics(message.decode())
        with self._lock:
            for side, block in parsed.items():
                if block is not None:
                    self._inputs.glove[side] = GloveSample(block, now)

    def _on_poses(self, message):
        now = time.monotonic()
        parsed = parse_board_poses(json.loads(message))
        with self._lock:
            for index, (pose, confidence, alternatives) in parsed.items():
                side = self._board_to_side.get(index)
                if side is not None:
                    self._inputs.wrist[side] = WristSample(pose, confidence, now, alternatives)

    def _on_skeleton(self, message):
        now = time.monotonic()
        parsed = parse_skeleton(message.decode())
        with self._lock:
            for glove, rotation in parsed.items():
                side = self._glove_to_side.get(glove)
                if side is not None:
                    self._inputs.imu[side] = ImuSample(rotation, now)

    def latest(self):
        with self._lock:
            return Inputs(dict(self._inputs.wrist), dict(self._inputs.glove), dict(self._inputs.imu))

    def close(self):
        for receiver in self._receivers:
            receiver.stop()


# -----------------------------------------------------------------------------
# No hardware
# -----------------------------------------------------------------------------
class MockSource:
    """Both Wrist boards trace slow loops with a gentle wrist twist, and both
    Gloves open and close into a fist every ``finger_period_s``."""

    def __init__(self, amplitude_m=0.06, twist_deg=20.0, finger_period_s=6.0):
        self.amplitude = amplitude_m
        self.twist = np.radians(twist_deg)
        self.finger_period = finger_period_s
        self.start = time.monotonic()
        self.origin = {"right": np.array([-0.25, 0.35, 0.25]), "left": np.array([0.25, 0.35, 0.25])}

    def sample_at(self, t):
        inputs = Inputs()
        stamp = self.start + t
        for side in SIDES:
            mirror = 1.0 if side == "right" else -1.0
            offset = self.amplitude * np.array([mirror * np.sin(0.5 * t), np.sin(0.35 * t), 0.5 * np.sin(0.7 * t)])
            rotation = R.from_euler("zx", [mirror * self.twist * np.sin(0.4 * t),
                                           0.5 * self.twist * np.sin(0.3 * t)]).as_matrix()
            inputs.wrist[side] = WristSample(Pose(self.origin[side] + offset, rotation), 1.0, stamp)
            curl = 0.5 - 0.5 * np.cos(2.0 * np.pi * t / self.finger_period)
            ergonomics = np.zeros(20)
            ergonomics[[5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]] = 75.0 * curl
            ergonomics[[1, 2, 3]] = 40.0 * curl
            inputs.glove[side] = GloveSample(ergonomics, stamp)
        return inputs

    def latest(self):
        return self.sample_at(time.monotonic() - self.start)

    def close(self):
        pass


class ReplaySource:
    """Plays back the raw inputs of a Demonstration at their recorded timing."""

    def __init__(self, npz_path, loop=False):
        data = np.load(npz_path)
        self.time = data["time"] - data["time"][0]
        self.data = {k: data[k] for k in data.files}
        self.loop = loop
        self.start = None          # the clock starts with the first latest()

    def finished(self):
        return (not self.loop and self.start is not None
                and time.monotonic() - self.start > self.time[-1])

    def latest(self):
        if self.start is None:
            self.start = time.monotonic()
        elapsed = time.monotonic() - self.start
        if self.loop:
            elapsed %= max(self.time[-1], 1e-6)
        i = int(np.clip(np.searchsorted(self.time, elapsed, side="right") - 1, 0, len(self.time) - 1))
        now = time.monotonic()
        inputs = Inputs(engaged={s: {"engaged": bool(self.data[f"{s}_engaged"][i]),
                                     "arm_q": self.data[f"{s}_arm_q_commanded"][i],
                                     "hand_q": self.data[f"{s}_hand_q_commanded"][i]} for s in SIDES})
        for side in SIDES:
            if self.data[f"{side}_wrist_valid"][i]:
                # The pose the session used (camera's choice, or the alternative the
                # IMU picked), so the replay does not depend on the IMU fit's timing.
                if f"{side}_wrist_used_alternative" in self.data and self.data[f"{side}_wrist_used_alternative"][i]:
                    pose = Pose(self.data[f"{side}_wrist_alt_position"][i], self.data[f"{side}_wrist_alt_rotation"][i])
                else:
                    pose = Pose(self.data[f"{side}_wrist_position"][i], self.data[f"{side}_wrist_rotation"][i])
                inputs.wrist[side] = WristSample(pose, float(self.data[f"{side}_wrist_confidence"][i]), now)
            if self.data[f"{side}_glove_valid"][i]:
                inputs.glove[side] = GloveSample(self.data[f"{side}_glove_ergonomics_deg"][i], now)
        return inputs

    def close(self):
        pass
