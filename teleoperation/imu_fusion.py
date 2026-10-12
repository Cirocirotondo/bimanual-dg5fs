"""Glove IMU resolves the two-fold ambiguity of a Wrist board seen by one tag.

When the camera sees a single small tag almost face-on, two board poses explain
the image about equally well (the estimator publishes both). The Glove's IMU
measures the hand orientation directly, so it can tell which one is right.

The Wrist board is rigidly strapped to the Glove, hence for the right pose
    R_world_board = A  R_imu  B
with two constant rotations: A (IMU reference frame -> World board frame; its
yaw differs per Glove and per MANUS session) and B (IMU -> Wrist board, a
property of how the board is strapped on: constant across sessions).

With an overhead camera the wrong pose is close to the true one turned by 180
deg about the vertical, which a rigidly attached IMU with free A and B could
explain just as well. So:

* A and B are fitted only on unambiguous samples (several tags, or a tag seen
  obliquely enough for the estimator to publish a single pose), and B is kept
  (saved by the caller) once known;
* with B known, A alone is fitted on all samples by consensus: the wrong
  branch is no longer consistent with the known B.

Until a fit is accepted the camera's own choice is kept. Fits can run in a
background thread (start()), so the control loop only pays for add()/choose().
"""
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


class ImuOrientationResolver:
    def __init__(self, buffer_s=20.0, refit_every_s=1.0, min_samples=60, min_trusted=30,
                 min_rotation_spread_deg=25.0, max_residual_deg=15.0, consensus_deg=12.0,
                 max_samples_in_fit=150, mount=None):
        """mount: known B (Rotation), e.g. saved from a previous session."""
        self.buffer_s = buffer_s
        self.refit_every_s = refit_every_s
        self.min_samples = min_samples
        self.min_trusted = min_trusted
        self.min_spread = np.radians(min_rotation_spread_deg)
        self.max_residual = np.radians(max_residual_deg)
        self.consensus = np.radians(consensus_deg)
        self.max_samples_in_fit = max_samples_in_fit
        self.buffer = deque()            # (stamp, candidate quats (k, 4), imu quat)
        self.A = None
        self.B = mount
        self.mount_from_fit = False      # B was fitted in this session
        self.residual_deg = None
        self.last_fit = None
        self._lock = threading.Lock()
        self._running = False

    @property
    def calibrated(self):
        return self.A is not None and self.B is not None

    # -- data in / decision out --------------------------------------------------
    def add(self, stamp, candidates, imu):
        """candidates: [Rotation] (camera's choice first; one if unambiguous); imu: Rotation."""
        with self._lock:
            self.buffer.append((stamp, np.array([c.as_quat() for c in candidates]), imu.as_quat()))
            while self.buffer and stamp - self.buffer[0][0] > self.buffer_s:
                self.buffer.popleft()

    def choose(self, candidates, imu):
        """Index of the candidate that agrees with the IMU (0 until calibrated)."""
        with self._lock:
            A, B = self.A, self.B
        if A is None or B is None or len(candidates) == 1:
            return 0
        predicted = A * imu * B
        return int(np.argmin([(c.inv() * predicted).magnitude() for c in candidates]))

    # -- background fitting ------------------------------------------------------
    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            time.sleep(self.refit_every_s)
            self.maybe_refit(time.monotonic(), force=True)

    def maybe_refit(self, now, force=False):
        """Refits (at most every refit_every_s unless forced). Returns True if a fit was accepted."""
        if not force and self.last_fit is not None and now - self.last_fit < self.refit_every_s:
            return False
        self.last_fit = now
        with self._lock:
            samples = list(self.buffer)
        trusted = [s for s in samples if len(s[1]) == 1]
        if len(trusted) >= self.min_trusted and self._spread(trusted) >= self.min_spread:
            A, B, residual = self._fit_both(self._subsample(trusted))
            if residual <= self.max_residual:
                with self._lock:
                    self.A, self.B, self.residual_deg = A, B, float(np.degrees(residual))
                    self.mount_from_fit = True
                return True
        if self.B is not None and len(samples) >= self.min_samples:
            A, residual = self._fit_yaw_reference(self._subsample(samples), self.B)
            if A is not None and residual <= self.max_residual:
                with self._lock:
                    self.A, self.residual_deg = A, float(np.degrees(residual))
                return True
        return False

    # -- fits ----------------------------------------------------------------------
    def _subsample(self, samples):
        if len(samples) <= self.max_samples_in_fit:
            return samples
        keep = np.linspace(0, len(samples) - 1, self.max_samples_in_fit).astype(int)
        return [samples[i] for i in keep]

    @staticmethod
    def _spread(samples):
        imu = R.from_quat(np.array([s[2] for s in samples]))
        return (imu * imu.mean().inv()).magnitude().max()

    @staticmethod
    def _fit_both(samples, iterations=30):
        board = R.from_quat(np.array([s[1][0] for s in samples]))
        imu = R.from_quat(np.array([s[2] for s in samples]))
        B = R.identity()
        for _ in range(iterations):
            A = (board * (imu * B).inv()).mean()
            B = ((A * imu).inv() * board).mean()
        residual = np.median((board.inv() * A * imu * B).magnitude())
        return A, B, residual

    def _fit_yaw_reference(self, samples, B):
        """A by consensus over every candidate of every sample, with B fixed."""
        imu = R.from_quat(np.array([s[2] for s in samples]))
        cand = np.array([[s[1][0], s[1][-1]] for s in samples])          # (N, 2, 4)
        hypotheses = [R.from_quat(cand[:, k]) * (imu * B).inv() for k in (0, 1)]
        pool = np.concatenate([h.as_quat() for h in hypotheses])
        best, best_count = None, -1
        for q in pool[:: max(1, len(pool) // 60)]:
            a = R.from_quat(q)
            d = np.minimum(*[(h.inv() * a).magnitude() for h in hypotheses])
            count = int((d < self.consensus).sum())
            if count > best_count:
                best, best_count = a, count
        if best_count < 0.6 * len(samples):
            return None, np.inf
        d = [(h.inv() * best).magnitude() for h in hypotheses]
        choice = (d[1] < d[0]).astype(int)
        inliers = np.minimum(d[0], d[1]) < self.consensus
        chosen = R.from_quat(cand[np.arange(len(samples)), choice])
        A = (chosen[inliers] * (imu[inliers] * B).inv()).mean()
        residual = np.median((chosen.inv() * A * imu * B).magnitude())
        return A, residual


def save_mount(path, B, residual_deg):
    Path(path).write_text(json.dumps({
        "description": "Glove IMU -> Wrist board rotation (B in R_world_board = A R_imu B)",
        "quaternion_xyzw": B.as_quat().tolist(),
        "fit_residual_deg": residual_deg,
    }, indent=2))


def load_mount(path):
    if not Path(path).exists():
        return None
    return R.from_quat(json.loads(Path(path).read_text())["quaternion_xyzw"])
