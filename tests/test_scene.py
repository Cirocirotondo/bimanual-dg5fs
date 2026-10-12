import mujoco
import numpy as np
import pytest

from bimanual_dg5fs import scene


@pytest.fixture(scope="module")
def model():
    return scene.build_model()


def _qadr(model, side, name):
    return model.joint(f"{side}/{name}").qposadr[0]


def _contact_bodies(model, data):
    return {tuple(sorted((model.body(model.geom_bodyid[c.geom1]).name,
                          model.body(model.geom_bodyid[c.geom2]).name)))
            for c in data.contact[: data.ncon]}


def test_every_limb_joint_has_a_position_actuator(model):
    expected = {f"{s}/{n}" for s in scene.SIDES for n in scene.ARM_JOINTS + scene.HAND_JOINTS}
    assert len(expected) == 52
    actuated = {model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu)}
    assert actuated == expected


def test_home_keyframe_is_reference_home_with_wrists_turned(model):
    key = model.key("home")
    right = np.array([key.qpos[_qadr(model, "right", n)] for n in scene.ARM_JOINTS])
    turn = np.zeros(6)
    turn[scene.ARM_JOINTS.index("wrist_2_joint")] = np.pi / 2
    turn[scene.ARM_JOINTS.index("wrist_3_joint")] = -np.pi
    np.testing.assert_allclose(right, np.array(scene.REFERENCE_HOME_ARM_Q_RIGHT) + turn)
    for side in scene.SIDES:
        hand = [key.qpos[_qadr(model, side, n)] for n in scene.HAND_JOINTS]
        np.testing.assert_array_equal(hand, 0.0)


def test_left_limb_mirrors_right_limb_at_home(model):
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    mirror = np.diag([-1.0, 1.0, 1.0])
    pairs = [(b, "left/" + model.body(b).name.split("/", 1)[1])
             for b in range(model.nbody) if model.body(b).name.startswith("right/")]
    assert len(pairs) > 30
    for right, left in pairs:
        # Tesollo's left and right DG-5F-S models are exact mirror images except
        # for the ring finger (finger 4), whose base sits 0.4 mm apart.
        tol = 5e-4 if left.split("/")[1].startswith("link_4_") else 1e-6
        np.testing.assert_allclose(data.body(left).xpos, mirror @ data.xpos[right], atol=tol,
                                   err_msg=left)


def test_hands_point_forward_with_thumbs_up_at_home(model):
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    for side in scene.SIDES:
        palm = data.body(f"{side}/link_base")
        assert palm.xmat.reshape(3, 3)[1, 2] < -0.95          # Hand axis along -y
        assert data.body(f"{side}/link_1_tip").xpos[2] > palm.xpos[2] + 0.05


def test_only_the_object_touches_anything_at_home(model):
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    assert _contact_bodies(model, data) == {("object", "table")}


def test_scene_holds_home(model):
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    q0 = data.qpos.copy()
    for _ in range(1000):  # 2 s
        mujoco.mj_step(model, data)
    robot = [_qadr(model, s, n) for s in scene.SIDES for n in scene.ARM_JOINTS + scene.HAND_JOINTS]
    assert np.abs(data.qpos[robot] - q0[robot]).max() < 1e-3
    obj = model.joint("object_free").qposadr[0]
    assert np.linalg.norm(data.qpos[obj:obj + 3] - q0[obj:obj + 3]) < 1e-3


@pytest.fixture(scope="module")
def contact_free_model():
    # Large single-joint moves run fingers into their neighbours or the Arm into
    # the Table; this checks the drives, so contacts are switched off.
    model = scene.build_model()
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return model


@pytest.mark.parametrize("side", scene.SIDES)
@pytest.mark.parametrize("name", scene.ARM_JOINTS + scene.HAND_JOINTS)
def test_each_joint_follows_its_actuator(contact_free_model, side, name):
    model = contact_free_model
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    joint = model.joint(f"{side}/{name}")
    act = model.actuator(f"{side}/{name}").id
    lo, hi = joint.range
    start = data.qpos[joint.qposadr[0]]
    # Move a third of the way towards the farther joint limit.
    target = start + (hi - start if hi - start >= start - lo else lo - start) / 3.0
    data.ctrl[act] = target
    for _ in range(500):  # 1 s
        mujoco.mj_step(model, data)
    moved = data.qpos[joint.qposadr[0]] - start
    assert moved / (target - start) > 0.9
