import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from bimanual_dg5fs import scene
from teleoperation.core import TeleopCore, load_config
from teleoperation.finger_retargeting import joint_limits_from_model
from teleoperation.inputs import GloveSample, Inputs, WristSample
from teleoperation.wrist_mapping import Pose

DT = 0.02


@pytest.fixture(scope="module")
def limits():
    model = scene.build_model()
    return {s: joint_limits_from_model(model, s) for s in scene.SIDES}


def _config_without_mounts():
    """The real config, without touching the saved Glove IMU mounts."""
    config = load_config()
    config["imu_fusion"]["mount_file"] = None
    return config


@pytest.fixture
def core(limits):
    return TeleopCore(_config_without_mounts(), limits, DT)


class Operator:
    """Feeds the core like the live sources would: one fresh sample per tick."""

    def __init__(self, core):
        self.core = core
        self.now = 100.0
        self.wrist = {"right": np.array([-0.25, 0.35, 0.25]), "left": np.array([0.25, 0.35, 0.25])}
        self.rotation = {s: np.eye(3) for s in scene.SIDES}
        self.ergonomics = {s: np.zeros(20) for s in scene.SIDES}
        self.tracked = {s: True for s in scene.SIDES}

    def inputs(self):
        inputs = Inputs()
        for side in scene.SIDES:
            if self.tracked[side]:
                inputs.wrist[side] = WristSample(Pose(self.wrist[side].copy(), self.rotation[side].copy()),
                                                 1.0, self.now)
            inputs.glove[side] = GloveSample(self.ergonomics[side].copy(), self.now)
        return inputs

    def tick(self, n=1):
        """Returns the last tick's targets and the messages of all n ticks."""
        messages = []
        for _ in range(n):
            self.now += DT
            targets, new = self.core.update(self.inputs(), self.now)
            messages += new
        return targets, messages


def test_engaging_needs_a_wrist_pose(core):
    assert core.engage("right") is not None
    operator = Operator(core)
    operator.tick()
    assert core.engage("right") is None and core.limbs["right"].engaged


def test_hand_base_follows_wrist_translation(core):
    operator = Operator(core)
    operator.tick()
    start = core.ik.hand_base_pose("right").position
    core.engage("right")
    operator.wrist["right"] = operator.wrist["right"] + [0.0, -0.04, 0.03]
    operator.tick(100)
    scale = core.config["mapping"]["position_scale"]
    np.testing.assert_allclose(core.ik.hand_base_pose("right").position, start + scale * np.array([0.0, -0.04, 0.03]),
                               atol=2e-3)


def test_hand_base_follows_wrist_rotation_about_itself(core):
    operator = Operator(core)
    operator.tick()
    start = core.ik.hand_base_pose("left")
    core.engage("left")
    turn = R.from_euler("z", 25, degrees=True).as_matrix()
    operator.rotation["left"] = turn
    operator.tick(100)
    pose = core.ik.hand_base_pose("left")
    # The Wrist calibration of the left Glove is identity, so wrist = board.
    np.testing.assert_allclose(pose.position, start.position, atol=2e-3)
    assert np.degrees(R.from_matrix(pose.rotation.T @ turn @ start.rotation).magnitude()) < 0.5


def test_clutched_limb_holds_and_reengages_without_jump(core):
    operator = Operator(core)
    operator.tick()
    core.engage("right")
    operator.wrist["right"] = operator.wrist["right"] + [0.03, 0.0, 0.0]
    operator.tick(50)
    core.clutch("right")
    held_q = core.ik.arm_q("right")
    held_pose = core.ik.hand_base_pose("right").position
    operator.wrist["right"] = operator.wrist["right"] + [0.0, 0.1, 0.0]   # Operator repositions
    operator.tick(50)
    np.testing.assert_array_equal(core.ik.arm_q("right"), held_q)
    core.engage("right")
    operator.tick(5)
    np.testing.assert_allclose(core.ik.hand_base_pose("right").position, held_pose, atol=1e-3)


def _home_hand_base(core, side):
    q = core.ik.arm_q(side)
    core.ik.set_arm_q(side, scene.HOME_ARM_Q[side])
    pose = core.ik.hand_base_pose(side)
    core.ik.set_arm_q(side, q)
    return pose


def test_waiting_limb_engages_after_continuous_visibility(core):
    operator = Operator(core)
    core.request_engage("right")
    operator.tracked["right"] = False
    operator.tick(int(5 / DT))
    assert not core.limbs["right"].engaged                        # never visible
    operator.tracked["right"] = True
    operator.tick(int(2 / DT))
    operator.tracked["right"] = False                            # lost before 3 s ...
    operator.tick(int(1 / DT))
    operator.tracked["right"] = True                             # ... so the countdown restarts
    operator.tick(int(2.5 / DT))
    assert not core.limbs["right"].engaged
    _, messages = operator.tick(int(1 / DT))
    assert core.limbs["right"].engaged and "right Limb engaged" in messages
    assert not core.limbs["left"].engaged                        # not asked to engage


def test_reengaging_pairs_the_new_wrist_pose_with_the_home_pose(core):
    operator = Operator(core)
    operator.tick()
    core.engage("right")
    operator.wrist["right"] = operator.wrist["right"] + [0.05, -0.08, 0.06]
    operator.ergonomics["right"][[5, 9, 13, 17]] = 60.0
    operator.tick(100)                                           # Limb away from Home, fingers curled
    operator.tracked["right"] = False
    operator.tick(int(2.2 / DT))                                 # longer than tracking.lost_after_s
    assert not core.limbs["right"].engaged
    operator.wrist["right"] = np.array([-0.4, 0.2, 0.35])        # gloves put back somewhere else
    operator.tracked["right"] = True
    targets, _ = operator.tick(int(6 / DT))                      # Home + 3 s of visibility
    assert core.limbs["right"].engaged
    np.testing.assert_allclose(core.ik.arm_q("right"), scene.HOME_ARM_Q["right"], atol=1e-3)
    assert targets["right"].hand_q[scene.HAND_JOINTS.index("joint_2_2")] == pytest.approx(np.radians(60.0))
    home = _home_hand_base(core, "right")
    np.testing.assert_allclose(core.limbs["right"].mapper.hand0.position, home.position, atol=1e-6)
    np.testing.assert_allclose(core.limbs["right"].mapper.wrist0.position,
                               core.limbs["right"].wrist.position, atol=1e-2)
    # Moving the wrist from the new Neutral pose moves the Hand base from Home by the scaled amount.
    operator.wrist["right"] = operator.wrist["right"] + [0.0, 0.04, 0.0]
    operator.tick(60)
    scale = core.config["mapping"]["position_scale"]
    np.testing.assert_allclose(core.ik.hand_base_pose("right").position, home.position + [0.0, 0.04 * scale, 0.0],
                               atol=3e-3)


def test_clutch_by_command_holds_until_engaged_by_command(core):
    operator = Operator(core)
    operator.tick()
    core.engage("left")
    operator.wrist["left"] = operator.wrist["left"] + [0.0, -0.05, 0.0]
    operator.tick(60)
    core.toggle("left")                                          # clutch
    held = core.ik.arm_q("left")
    operator.tick(int(5 / DT))
    np.testing.assert_array_equal(core.ik.arm_q("left"), held)   # holds still, no auto re-engage
    assert not core.limbs["left"].engaged
    core.toggle("left")                                          # engage: Home first, then countdown
    operator.tick(int(6 / DT))
    assert core.limbs["left"].engaged
    np.testing.assert_allclose(core.ik.arm_q("left"), scene.HOME_ARM_Q["left"], atol=1e-3)


def test_glove_imu_overrides_wrong_camera_choice(limits):
    from scipy.spatial.transform import Rotation as R
    from teleoperation.inputs import ImuSample
    core = TeleopCore(_config_without_mounts(), limits, DT)
    limb = core.limbs["right"]
    A, B = R.random(random_state=3), R.random(random_state=4)
    rng = np.random.default_rng(0)
    board = R.identity()
    now = 100.0
    wrong_first = agree = checked = 0
    from tests.teleoperation.test_imu_fusion import mirrored
    board = R.from_euler("xy", [25, -15], degrees=True)
    for i in range(400):
        now += DT
        board = R.from_rotvec(rng.normal(scale=0.05, size=3)) * board
        tilt = np.degrees(np.arccos(board.apply([0, 0, 1])[2]))
        if tilt > 60:
            board = R.from_euler("x", -0.3 * (tilt - 60), degrees=True) * board
        wrong = mirrored(board)
        inputs = Inputs()
        if i % 3 == 0:                                   # unambiguous frame
            first_wrong = False
            inputs.wrist["right"] = WristSample(Pose(np.zeros(3), board.as_matrix()), 1.0, now)
        else:
            first_wrong = bool(i % 2)
            poses = [Pose(np.zeros(3), r.as_matrix()) for r in ([wrong, board] if first_wrong else [board, wrong])]
            inputs.wrist["right"] = WristSample(poses[0], 1.0, now, [poses[1]])
        inputs.imu["right"] = ImuSample(A.inv() * board * B.inv(), now)
        core.observe(inputs)
        limb.resolver.maybe_refit(now)
        if limb.resolver.calibrated and i > 300:
            agree += limb.used_alternative == first_wrong
            checked += 1
            wrong_first += first_wrong
    assert limb.resolver.calibrated and wrong_first > 30 and agree / checked > 0.97


def test_short_tracking_gap_holds_and_resumes_without_going_home(core):
    operator = Operator(core)
    operator.tick()
    core.engage("right")
    operator.wrist["right"] = operator.wrist["right"] + [0.03, 0.0, 0.0]
    operator.tick(50)
    neutral = core.limbs["right"].mapper.wrist0.position.copy()
    held = core.ik.arm_q("right")
    operator.tracked["right"] = False
    _, messages = operator.tick(int(1.5 / DT))                   # 1.5 s without the Wrist board
    assert core.limbs["right"].engaged and any("holds still" in m for m in messages)
    np.testing.assert_allclose(core.ik.arm_q("right"), held, atol=1e-3)
    operator.tracked["right"] = True
    _, messages = operator.tick(5)
    assert core.limbs["right"].engaged and any("continues" in m for m in messages)
    np.testing.assert_array_equal(core.limbs["right"].mapper.wrist0.position, neutral)   # same Neutral pose


def test_losing_the_wrist_board_clutches_the_limb(core):
    operator = Operator(core)
    operator.tick()
    core.engage("right")
    operator.tracked["right"] = False
    _, messages = operator.tick(int(2.2 / DT))
    assert not core.limbs["right"].engaged
    assert any("lost" in m for m in messages)


def test_fingers_follow_glove_only_while_engaged(core):
    operator = Operator(core)
    operator.ergonomics["right"][[5, 9, 13, 17]] = 60.0
    targets, _ = operator.tick()
    assert np.all(targets["right"].hand_q == 0.0)
    core.engage("right")
    targets, _ = operator.tick()
    assert targets["right"].hand_q[scene.HAND_JOINTS.index("joint_2_2")] == pytest.approx(np.radians(60.0))


def test_home_drives_arms_back(core):
    operator = Operator(core)
    operator.tick()
    core.engage("right")
    operator.wrist["right"] = operator.wrist["right"] + [0.05, -0.05, 0.05]
    operator.tick(100)
    core.home()
    operator.tick(200)
    np.testing.assert_allclose(core.ik.arm_q("right"), scene.HOME_ARM_Q["right"], atol=1e-4)
    assert not core.limbs["right"].engaged
