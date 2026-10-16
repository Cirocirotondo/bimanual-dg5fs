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


@pytest.fixture(scope="module")
def basin_model():
    return scene.build_model("basin")


def test_basin_rests_on_the_table_clear_of_the_hands(basin_model):
    model = basin_model
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    assert _contact_bodies(model, data) == {("object", "table")}
    assert model.body("object").mass[0] == pytest.approx(scene.BASIN_MASS)
    obj = model.joint("object_free").qposadr[0]
    q0 = data.qpos[obj:obj + 7].copy()
    for _ in range(1000):  # 2 s
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos[obj:obj + 7], q0, atol=1e-4)


def test_basin_has_one_handle_towards_each_hand(basin_model):
    model = basin_model
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    body = model.body("object").id
    bars = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            and abs(data.geom_xmat[g].reshape(3, 3)[1, 2]) > 0.99          # axis along y
            and abs(data.geom_xpos[g][1] - scene.BASIN_XY[1]) < 1e-6]
    assert len(bars) == 2
    for g in bars:
        side = "left" if data.geom_xpos[g][0] > 0 else "right"
        assert np.sign(data.body(f"{side}/link_base").xpos[0]) == np.sign(data.geom_xpos[g][0])


@pytest.fixture(scope="module")
def book_model():
    return scene.build_model("book")


def test_book_rests_closed_on_the_table(book_model):
    model = book_model
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    assert _contact_bodies(model, data) == {("object", "table")}
    obj = model.joint("object_free").qposadr[0]
    hinge = model.joint("book_cover_hinge").qposadr[0]
    q0 = data.qpos[obj:obj + 7].copy()
    for _ in range(500):  # 1 s: it settles into the soft contact, its mass a little towards the spine
        mujoco.mj_step(model, data)
    settled = data.qpos[obj:obj + 7].copy()
    np.testing.assert_allclose(settled, q0, atol=5e-4)       # < 0.5 mm, < 0.06 deg
    for _ in range(1000):  # 2 s more: no drift
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos[obj:obj + 7], settled, atol=1e-5)
    assert abs(data.qpos[hinge]) < np.deg2rad(0.5)       # shut on the (soft) joint limit
    book = [g for g in range(model.ngeom) if model.geom(g).name.startswith("book_")]
    np.testing.assert_allclose(model.geom_friction[book], np.tile(scene.BOOK_FRICTION, (len(book), 1)))


def test_book_cover_opens_over_the_spine_and_stays_open(book_model):
    model = book_model
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    hinge = model.joint("book_cover_hinge")
    cover = model.body("book_cover").id
    # Lift the cover past vertical with an external torque about the hinge, then let go.
    for _ in range(1500):  # 3 s
        data.qfrc_applied[hinge.dofadr[0]] = 0.15 if data.qpos[hinge.qposadr[0]] < np.deg2rad(100) else 0.0
        mujoco.mj_step(model, data)
    data.qfrc_applied[:] = 0.0
    for _ in range(1500):
        mujoco.mj_step(model, data)
    angle = np.rad2deg(data.qpos[hinge.qposadr[0]])
    # Past vertical it falls open over the spine (+x, the left Hand's side) and rests on the Table.
    assert 175.0 < angle <= scene.BOOK_HINGE_RANGE_DEG[1] + 1.0
    assert data.xipos[cover][0] > scene.BOOK_XY[0] + scene.BOOK_WIDTH / 2.0
    assert np.sign(data.body("left/link_base").xpos[0]) == np.sign(data.xipos[cover][0])


def test_book_model_is_the_scene_book(book_model):
    """The book alone (what the training stack rebuilds it from) has the Scene book's bodies,
    geoms and hinge, in the same frames (only the free base is not placed on the Table)."""
    alone = scene.book_model()
    assert [alone.body(b).name for b in range(1, alone.nbody)] == ["object", "book_cover"]
    for name in ("object", "book_cover"):
        a, s = alone.body(name), book_model.body(name)
        for field in ("mass", "inertia", "ipos", "iquat") + (("pos", "quat") if name == "book_cover" else ()):
            np.testing.assert_allclose(getattr(a, field), getattr(s, field), atol=1e-12, err_msg=f"{name}.{field}")
    geoms = [alone.geom(g).name for g in range(alone.ngeom)]
    assert geoms == [n for n in (book_model.geom(g).name for g in range(book_model.ngeom)) if n.startswith("book_")]
    for name in geoms:
        a, s = alone.geom(name), book_model.geom(name)
        for field in ("type", "size", "pos", "quat", "friction"):
            np.testing.assert_allclose(getattr(a, field), getattr(s, field), err_msg=f"{name}.{field}")
    a, s = alone.joint("book_cover_hinge"), book_model.joint("book_cover_hinge")
    for field in ("range", "axis", "pos", "damping", "frictionloss", "armature"):
        np.testing.assert_allclose(getattr(a, field), getattr(s, field), err_msg=f"hinge.{field}")
    np.testing.assert_allclose(a.range, np.deg2rad(scene.BOOK_HINGE_RANGE_DEG))


def test_object_joint_ranges_in_radians():
    assert scene.object_joint_ranges("cube") == {}
    assert scene.object_joint_ranges("basin") == {}
    (name, (lower, upper)), = scene.object_joint_ranges("book").items()
    assert name == "book_cover_hinge"
    np.testing.assert_allclose((lower, upper), np.deg2rad(scene.BOOK_HINGE_RANGE_DEG))


def test_basin_cube_scene_has_both_resting_apart_cube_nearer_the_bases():
    model = scene.build_model("basin_cube")
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    assert _contact_bodies(model, data) == {("object", "table"), ("cube", "table")}
    basin, cube = data.body("object").xpos.copy(), data.body("cube").xpos.copy()
    np.testing.assert_allclose(basin[:2], scene.BASIN_XY)
    assert abs(cube[1]) < abs(basin[1])                     # the bases are at y = 0
    rim_gap = np.linalg.norm(cube[:2] - basin[:2]) - scene.BASIN_TOP_RADIUS - scene.OBJECT_SIZE / 2.0
    assert rim_gap == pytest.approx(scene.BASIN_CUBE_GAP)
    adr = {n: model.joint(f"{n}_free").qposadr[0] for n in ("object", "cube")}
    q0 = {n: data.qpos[a:a + 7].copy() for n, a in adr.items()}
    for _ in range(500):  # 1 s: both settle into the soft contact
        mujoco.mj_step(model, data)
    settled = {n: data.qpos[a:a + 7].copy() for n, a in adr.items()}
    for n in adr:
        np.testing.assert_allclose(settled[n], q0[n], atol=5e-4)
    for _ in range(1000):  # 2 s more: no drift
        mujoco.mj_step(model, data)
    for n, a in adr.items():
        np.testing.assert_allclose(data.qpos[a:a + 7], settled[n], atol=1e-5)


def test_cube_fits_in_the_basin():
    inner_bottom = 2.0 * (scene.BASIN_BOTTOM_RADIUS - scene.BASIN_WALL)
    assert scene.OBJECT_SIZE * np.sqrt(2.0) < inner_bottom          # flat on the bottom, any yaw
    assert scene.OBJECT_SIZE < scene.BASIN_HEIGHT + 0.02             # about level with the rim


def _bag_held_by_its_ends(sealed=False):
    """The bag alone, lifted by its two end segments the way two Hands hold it: each end segment
    is welded to a carriage that keeps it level and at its height but slides freely along the
    bag's length, so the ends can come closer as the bag sags (rigid segments held at their
    full span could not sag at all)."""
    spec = mujoco.MjSpec()
    spec.compiler.autolimits = True
    height = 0.5
    body = spec.worldbody.add_body(name="object", pos=(0.0, 0.0, height))
    body.add_freejoint(name="object_free")
    scene._add_bag(spec, body, sealed)
    segment = scene.BAG_SIZE[0] / scene.BAG_SEGMENTS
    centre_x = scene.BAG_SIZE[0] / 2.0 - segment / 2.0
    for end, sign in (("bag_segment_0", -1.0), (f"bag_segment_{scene.BAG_SEGMENTS - 1}", 1.0)):
        carriage = spec.worldbody.add_body(name=f"carriage_{end}",
                                           pos=(sign * centre_x, 0.0, height + scene.BAG_SIZE[2] / 2.0))
        carriage.add_joint(name=f"slide_{end}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(1.0, 0.0, 0.0))
        carriage.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.005, 0.0, 0.0), mass=0.01,
                          contype=0, conaffinity=0)
        weld = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                 name1=end, name2=carriage.name)
        weld.data[:] = 0.0                        # anchor at the carriage, relative pose from qpos0
        weld.data[10] = 1.0                       # torquescale: hold the orientation as well
    return spec.compile()


@pytest.mark.parametrize("name", ["bag", "bag_sealed"])
def test_bag_rests_flat_on_the_table(name):
    model = scene.build_model(name)
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    bag_bodies = {model.body(b).name for b in range(model.nbody)
                  if model.body(b).name == "object" or model.body(b).name.startswith("bag_")}
    assert {b for pair in _contact_bodies(model, data) for b in pair} <= bag_bodies | {"table"}
    total = sum(model.body_mass[model.body(b).id] for b in bag_bodies)
    assert total == pytest.approx(scene.BAG_MASS)
    adr = model.joint("object_free").qposadr[0]
    for _ in range(500):  # 1 s: settles into the (hard) Table contact
        mujoco.mj_step(model, data)
    settled = data.qpos[adr:adr + 7].copy()
    for _ in range(1000):  # 2 s more: no drift, hardly any bending
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos[adr:adr + 7], settled, atol=5e-5)
    joints = [data.qpos[model.joint(j).qposadr[0]] for j in scene.OBJECT_JOINTS[name]]
    assert np.abs(joints).max() < np.deg2rad(0.5)        # the plain bag's raised ends droop by ~0.2 deg


def test_bag_held_by_its_ends_sags_by_the_calibrated_amount():
    model = _bag_held_by_its_ends()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    centre = model.body(f"bag_segment_{scene.BAG_SEGMENTS // 2}").id
    z0 = data.xpos[centre][2]
    for _ in range(1500):  # 3 s
        mujoco.mj_step(model, data)
    settled = data.xpos[centre][2]
    for _ in range(1500):  # 3 s more
        mujoco.mj_step(model, data)
    assert z0 - data.xpos[centre][2] == pytest.approx(scene.BAG_SAG, abs=0.005)
    assert abs(data.xpos[centre][2] - settled) < 5e-4                         # damped: settled, not ringing
    bends = np.rad2deg([data.qpos[model.joint(f"bag_bend_{j}").qposadr[0]] for j in range(scene.BAG_SEGMENTS - 1)])
    assert np.abs(bends).max() < 0.75 * scene.BAG_BEND_RANGE_DEG[1]          # a curve, not a kink on a limit


def test_a_pressing_fingertip_sinks_into_the_bag_by_the_calibrated_amount():
    force = 10.0
    spec = scene.build_spec("bag")
    segment = scene.BAG_SIZE[0] / scene.BAG_SEGMENTS
    top = scene.TABLE_TOP_Z + scene.BAG_SIZE[2]
    probe = spec.worldbody.add_body(name="probe", pos=(scene.BAG_XY[0] - segment / 2.0, scene.BAG_XY[1], top + 0.0101))
    probe.add_freejoint(name="probe_free")
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.01, 0.0, 0.0), mass=force / 9.81)
    model = spec.compile()
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    adr = model.joint("probe_free").qposadr[0]
    data.qpos[adr:adr + 7] = (*model.body("probe").pos, 1.0, 0.0, 0.0, 0.0)
    for _ in range(2000):  # 4 s
        mujoco.mj_step(model, data)
    root = model.geom(f"bag_{scene.BAG_SEGMENTS // 2 - 1}").id
    surface = data.geom_xpos[root][2] + scene.BAG_SIZE[2] / 2.0
    assert surface - (data.qpos[adr + 2] - 0.01) == pytest.approx(scene.BAG_SQUEEZE, abs=0.0015)


def test_sealed_bag_has_a_flat_seam_beyond_each_short_end_towards_each_hand():
    model = scene.build_model("bag_sealed")
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    seams = [g for g in range(model.ngeom) if model.geom(g).name.startswith("bag_seam_")]
    assert len(seams) == 2
    for g in seams:
        np.testing.assert_allclose(2.0 * model.geom_size[g], scene.BAG_SEAM)
        assert abs(data.geom_xpos[g][0] - scene.BAG_XY[0]) == pytest.approx(
            scene.BAG_SIZE[0] / 2.0 + scene.BAG_END_ROUND + scene.BAG_SEAM[0] / 2.0)
        side = "left" if data.geom_xpos[g][0] > 0 else "right"
        assert np.sign(data.body(f"{side}/link_base").xpos[0]) == np.sign(data.geom_xpos[g][0])
    plain = scene.build_model("bag")
    assert not any(plain.geom(g).name.startswith("bag_seam_") for g in range(plain.ngeom))


def test_bag_model_matches_the_scene_bag():
    alone, in_scene = scene.bag_model(sealed=True), scene.build_model("bag_sealed")
    for name in scene.OBJECT_JOINTS["bag_sealed"]:
        np.testing.assert_allclose(alone.joint(name).range, in_scene.joint(name).range)
        assert alone.jnt_stiffness[alone.joint(name).id] == in_scene.jnt_stiffness[in_scene.joint(name).id]
    assert scene.object_joint_ranges("bag")["bag_bend_0"][1] == pytest.approx(np.deg2rad(scene.BAG_BEND_RANGE_DEG[1]))


def test_plain_bag_leaves_room_under_its_ends_for_the_fingers():
    model = scene.build_model("bag")
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    for side in ("left", "right"):
        lowest = []
        for name in (f"bag_end_{side}", f"bag_{0 if side == 'right' else scene.BAG_SEGMENTS - 1}"):
            g = model.geom(name).id
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start + count] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            outer = np.abs(vertices[:, 0] - scene.BAG_XY[0]) >= scene.BAG_SIZE[0] / 2.0 - 1e-6
            lowest.append(vertices[outer, 2].min())
        assert min(lowest) - scene.TABLE_TOP_Z == pytest.approx(scene.BAG_END_LIFT, abs=1e-6)
