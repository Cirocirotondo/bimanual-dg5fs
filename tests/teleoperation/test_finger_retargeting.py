import mujoco
import numpy as np
import pytest

from bimanual_dg5fs import scene
from teleoperation.finger_retargeting import (HAND_JOINTS, FingerRetargeter, joint_limits_from_model,
                                              tesollo_targets)

FLEX = [5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]   # MCP/PIP/DIP stretch of fingers 2..5


@pytest.fixture(scope="module")
def model():
    return scene.build_model()


def test_open_glove_matches_tesollo_formula():
    # Relaxed Glove (all angles 0): thumb 1_1 = 58.5 deg, 1_2 = 20 deg * 1.6, rest 0.
    right = tesollo_targets("right", np.zeros(20))
    left = tesollo_targets("left", np.zeros(20))
    np.testing.assert_allclose(right[:2], np.radians([58.5, -32.0]))
    np.testing.assert_allclose(left[:2], np.radians([-58.5, 32.0]))
    np.testing.assert_array_equal(right[2:], 0.0)
    np.testing.assert_array_equal(left[2:], 0.0)


def test_pinky_palm_fold_rule():
    q = np.zeros(20)
    q[16], q[17], q[18] = 12.0, 60.0, 30.0           # curled pinky: |spread| * 2
    assert tesollo_targets("right", q)[16] == pytest.approx(np.radians(24.0))
    q[17] = 10.0                                      # otherwise |spread| / 1.5
    assert tesollo_targets("right", q)[16] == pytest.approx(np.radians(8.0))


@pytest.mark.parametrize("side", scene.SIDES)
def test_output_stays_within_hand_limits(model, side):
    retarget = FingerRetargeter(side, joint_limits_from_model(model, side))
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = retarget(rng.uniform(-90, 120, 20))
        assert np.all(q >= retarget.lower - 1e-12) and np.all(q <= retarget.upper + 1e-12)


def _fingertips(model, data, ergonomics):
    """World fingertip positions and the same in each Hand-base frame, at Home."""
    scene.reset_to_home(model, data)
    for side in scene.SIDES:
        q = FingerRetargeter(side, joint_limits_from_model(model, side))(ergonomics)
        for name, value in zip(HAND_JOINTS, q):
            data.qpos[model.joint(f"{side}/{name}").qposadr[0]] = value
    mujoco.mj_kinematics(model, data)
    world, local = {}, {}
    for side in scene.SIDES:
        base = data.body(f"{side}/link_base")
        rot = base.xmat.reshape(3, 3)
        world[side] = np.array([data.body(f"{side}/link_{f}_tip").xpos for f in range(1, 6)])
        local[side] = (world[side] - base.xpos) @ rot
    return world, local


def test_same_glove_angles_give_mirrored_hands(model):
    # At Home the Limbs mirror each other across x = 0, so the left formula must
    # produce the mirror image of the right one (Tesollo's ring finger is 0.4 mm off).
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1)
    for _ in range(5):
        world, _ = _fingertips(model, data, rng.uniform(-20, 80, 20))
        np.testing.assert_allclose(world["left"], world["right"] * [-1, 1, 1], atol=5e-4)


def test_fist_curls_fingers_towards_the_palm(model):
    data = mujoco.MjData(model)
    _, open_tips = _fingertips(model, data, np.zeros(20))
    fist = np.zeros(20)
    fist[FLEX] = 75.0
    _, fist_tips = _fingertips(model, data, fist)
    for side in scene.SIDES:
        # Hand-base frame: z runs along the fingers, +x is the palm side (where the thumb sits).
        assert np.all(open_tips[side][0, 0] > 0.05)
        assert np.all(fist_tips[side][1:, 2] < open_tips[side][1:, 2] - 0.05)
        assert np.all(fist_tips[side][1:, 0] > open_tips[side][1:, 0] + 0.02)
