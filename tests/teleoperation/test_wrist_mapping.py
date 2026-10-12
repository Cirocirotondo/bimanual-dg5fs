import numpy as np
from scipy.spatial.transform import Rotation as R

from teleoperation.core import TELEOP_DIR
from teleoperation.wrist_mapping import (Pose, RelativeMapper, WristCalibration, WristPoseFilter,
                                         rate_limit, rotation_angle_deg)


def _pose(xyz, rotvec=(0.0, 0.0, 0.0)):
    return Pose(np.asarray(xyz, dtype=float), R.from_rotvec(rotvec).as_matrix())


def test_filter_rejects_jumps_and_blends():
    f = WristPoseFilter(translation_alpha=0.7, rotation_alpha=0.85, max_jump_m=0.08, max_jump_deg=35.0)
    first = f.update(_pose([0, 0, 0]))
    np.testing.assert_allclose(first.position, 0.0)
    assert f.update(_pose([0.1, 0, 0])) is None                      # 10 cm jump
    assert f.update(_pose([0, 0, 0], [0, 0, np.radians(40)])) is None  # 40 deg jump
    blended = f.update(_pose([0.01, 0, 0], [0, 0, np.radians(10)]))
    np.testing.assert_allclose(blended.position, [0.007, 0, 0])
    assert abs(rotation_angle_deg(np.eye(3), blended.rotation) - 8.5) < 1e-6


def test_filter_reacquires_after_starting_on_an_outlier():
    f = WristPoseFilter(0.7, 0.85, 0.08, 35.0, reacquire_after=5)
    f.update(_pose([2.0, 0, 0]))                       # a wild first sample
    results = [f.update(_pose([0, 0, 0.001 * i])) for i in range(6)]
    assert all(r is None for r in results[:4])
    assert results[4] is not None and np.linalg.norm(results[4].position) < 0.01
    assert results[5] is not None                      # tracking again


def test_filter_ignores_isolated_outliers():
    f = WristPoseFilter(0.7, 0.85, 0.08, 35.0, reacquire_after=5)
    f.update(_pose([0, 0, 0]))
    for i in range(20):                                # a different wild pose every other frame
        assert f.update(_pose([1.0 + i, 0, 0])) is None
        assert f.update(_pose([0, 0, 0])) is not None


def test_wrist_calibration_is_applied_in_metres():
    calibration = WristCalibration.from_file(TELEOP_DIR / "calibration" / "right_wrist2board_calibration.json")
    offset = np.array([-0.045489509630624474, 0.05457155810851333, -0.025233587847748228])
    board = _pose([1.0, 2.0, 3.0], [0, 0, np.pi / 2])
    wrist = calibration.wrist_pose(board)
    np.testing.assert_allclose(wrist.position, board.position + board.rotation @ offset)
    assert 0.07 < np.linalg.norm(wrist.position - board.position) < 0.08


def test_relative_mapping_moves_hand_by_wrist_motion():
    mapper = RelativeMapper()
    hand0 = _pose([0.3, -0.6, 0.25], [0.3, -0.2, 1.0])
    wrist0 = _pose([-0.2, 0.4, 0.3], [1.0, 0.5, 0.0])
    mapper.engage(wrist0, hand0)

    at_neutral = mapper.target(wrist0)
    np.testing.assert_allclose(at_neutral.position, hand0.position)
    np.testing.assert_allclose(at_neutral.rotation, hand0.rotation, atol=1e-12)

    moved = mapper.target(Pose(wrist0.position + [0.02, -0.03, 0.05], wrist0.rotation))
    np.testing.assert_allclose(moved.position, hand0.position + [0.02, -0.03, 0.05])

    # Turning the wrist 30 deg about world z turns the Hand base 30 deg about world z, in place.
    turn = R.from_euler("z", 30, degrees=True).as_matrix()
    turned = mapper.target(Pose(wrist0.position, turn @ wrist0.rotation))
    np.testing.assert_allclose(turned.position, hand0.position)
    np.testing.assert_allclose(turned.rotation, turn @ hand0.rotation, atol=1e-12)


def test_position_scale_amplifies_translation_not_rotation():
    mapper = RelativeMapper(scale=1.5)
    hand0 = _pose([0.3, -0.6, 0.25])
    wrist0 = _pose([-0.2, 0.4, 0.3])
    mapper.engage(wrist0, hand0)
    turn = R.from_euler("x", 20, degrees=True).as_matrix()
    target = mapper.target(Pose(wrist0.position + [0.10, 0.0, 0.0], turn @ wrist0.rotation))
    np.testing.assert_allclose(target.position, hand0.position + [0.15, 0.0, 0.0])   # 10 cm -> 15 cm
    np.testing.assert_allclose(target.rotation, turn @ hand0.rotation, atol=1e-12)     # rotation 1:1


def test_rate_limit_caps_each_step():
    previous = _pose([0, 0, 0])
    target = _pose([1.0, 0, 0], [0, 0, np.pi / 2])
    step = rate_limit(previous, target, dt=0.02, max_speed_m_s=0.3, max_rotation_speed_deg_s=180.0)
    np.testing.assert_allclose(step.position, [0.006, 0, 0])
    assert abs(rotation_angle_deg(previous.rotation, step.rotation) - 3.6) < 1e-6
