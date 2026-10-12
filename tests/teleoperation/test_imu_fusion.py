import numpy as np
from scipy.spatial.transform import Rotation as R

from teleoperation.imu_fusion import ImuOrientationResolver


def mirrored(board, view=np.array([0.0, 0.0, -1.0])):
    """The other IPPE solution of a single tag: its normal (board z) reflected about
    the camera's line of sight (here an overhead camera looking down)."""
    normal = board.apply([0.0, 0.0, 1.0])
    towards_camera = -view
    axis = np.cross(normal, towards_camera)
    if np.linalg.norm(axis) < 1e-9:
        return board
    angle = 2.0 * np.arccos(np.clip(normal @ towards_camera, -1.0, 1.0))
    return R.from_rotvec(axis / np.linalg.norm(axis) * angle) * board


def _scenario(n=300, seed=0, trusted_every=3, A=None, B=None):
    """A wrist that keeps turning while its tag faces an overhead camera. Every
    trusted_every-th frame is unambiguous (one candidate); otherwise the camera
    reports the true and the mirrored pose, in random order."""
    rng = np.random.default_rng(seed)
    A = R.random(random_state=1) if A is None else A
    B = R.random(random_state=2) if B is None else B
    board = R.from_euler("xy", [25, -15], degrees=True)
    samples = []
    for i in range(n):
        board = R.from_rotvec(rng.normal(scale=0.05, size=3)) * board
        tilt = np.degrees(np.arccos(board.apply([0, 0, 1])[2]))
        if tilt > 60:                                    # keep the tag visible from above
            board = R.from_euler("x", -0.3 * (tilt - 60), degrees=True) * board
        imu = A.inv() * board * B.inv() * R.from_rotvec(rng.normal(scale=0.01, size=3))
        if trusted_every and i % trusted_every == 0:
            samples.append((i / 30.0, [board], imu, 0))
            continue
        camera_first_wrong = bool(rng.integers(0, 2))
        wrong = mirrored(board)
        candidates = [wrong, board] if camera_first_wrong else [board, wrong]
        samples.append((i / 30.0, candidates, imu, int(camera_first_wrong)))
    return samples


def _accuracy(resolver, samples):
    correct = decided = 0
    for t, candidates, imu, true_index in samples:
        resolver.add(t, candidates, imu)
        resolver.maybe_refit(t)
        if resolver.calibrated and len(candidates) > 1:
            decided += 1
            correct += resolver.choose(candidates, imu) == true_index
    return correct, decided


def test_resolver_learns_from_unambiguous_frames_and_picks_the_true_pose():
    resolver = ImuOrientationResolver()
    correct, decided = _accuracy(resolver, _scenario())
    assert resolver.calibrated and resolver.mount_from_fit and resolver.residual_deg < 5.0
    assert decided > 100 and correct / decided > 0.99


def test_known_mount_resolves_a_session_with_only_ambiguous_frames():
    B = R.random(random_state=2)
    resolver = ImuOrientationResolver(mount=B)
    # a new MANUS session: different IMU yaw reference, and no unambiguous frame at all
    samples = _scenario(seed=5, trusted_every=0, A=R.from_euler("z", 140, degrees=True) * R.random(random_state=1), B=B)
    correct, decided = _accuracy(resolver, samples)
    assert resolver.calibrated and not resolver.mount_from_fit
    assert decided > 150 and correct / decided > 0.99


def test_without_mount_ambiguous_frames_alone_never_calibrate():
    resolver = ImuOrientationResolver()
    correct, decided = _accuracy(resolver, _scenario(trusted_every=0))
    assert not resolver.calibrated and decided == 0


def test_resolver_keeps_the_camera_choice_until_the_wrist_has_turned():
    resolver = ImuOrientationResolver(min_trusted=10)
    still = R.from_euler("x", 10, degrees=True)
    for i in range(200):
        resolver.add(i / 30.0, [still], R.identity())
        resolver.maybe_refit(i / 30.0)
    assert not resolver.calibrated
