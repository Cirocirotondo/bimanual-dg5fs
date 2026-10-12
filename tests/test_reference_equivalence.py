"""The Twin's Arms move exactly like the Reference setup's Arms.

At the same joint angles, each Twin flange sits where the Reference flange sits
relative to its Arm base. The Base poses themselves differ slightly, because the
Twin's are made symmetric. The Reference model (tests/fixtures/reference_scene, copied from
robohand-master_wsl3) is the MuJoCo Menagerie UR5e; the Twin uses the official
UR URDF. Both are compared at the same joint angles.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from bimanual_dg5fs import scene

REFERENCE_SCENE = Path(__file__).parent / "fixtures" / "reference_scene" / "bimanual_scene_cube.xml"
REFERENCE_PREFIX = {side: f"{arm}/" for side, arm in scene.REFERENCE_ARM.items()}

# Menagerie rounds the UR5e link lengths (0.163 / 0.392 / 0.1 / 0.1 m) where
# the official URDF uses UR's nominal values (0.1625 / 0.3922 / 0.0997 / 0.0996 m),
# which puts the flanges 1.1-1.3 mm apart at the same joint angles.
FLANGE_POS_TOL = 2e-3      # m
ANGLE_TOL_DEG = 0.1
# Twin tool0 = Reference wrist_3_link frame rotated -90 deg about x (z out of the flange).
TOOL0_IN_REFERENCE_WRIST3 = R.from_euler("x", -90, degrees=True)
# The Reference flange face is 0.1 m along wrist_3_link's y axis (where its placeholder box sits).
REFERENCE_FLANGE_OFFSET = np.array([0.0, 0.1, 0.0])


@pytest.fixture(scope="module")
def reference():
    spec = mujoco.MjSpec.from_file(str(REFERENCE_SCENE))
    # Kinematics only: drop the 30 MB of visual meshes, which are not in the fixture.
    for geom in list(spec.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(geom)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    model = spec.compile()
    return model, mujoco.MjData(model)


@pytest.fixture(scope="module")
def twin():
    model = scene.build_model()
    return model, mujoco.MjData(model)


def _set_arms(model, data, q, joint_name):
    for side in scene.SIDES:
        for name, value in zip(scene.ARM_JOINTS, q[side]):
            data.qpos[model.joint(joint_name(side, name)).qposadr[0]] = value
    mujoco.mj_forward(model, data)


def _angle_deg(rot_a, rot_b):
    return np.degrees((R.from_matrix(rot_a).inv() * R.from_matrix(rot_b)).magnitude())


@pytest.mark.parametrize("side", scene.SIDES)
def test_base_frame_matches_reference_base_pose(reference, twin, side):
    """Same orientation (this guards the URDF base_link vs base yaw); the
    positions differ only by the symmetrisation of the Base poses (< 2.5 cm)."""
    ref_model, ref_data = reference
    twin_model, twin_data = twin
    mujoco.mj_forward(ref_model, ref_data)
    mujoco.mj_forward(twin_model, twin_data)
    ref_base = ref_data.body(REFERENCE_PREFIX[side][:-1] + "_")  # ur5_1_ / ur5_2_
    twin_base = twin_data.body(f"{side}/base")
    assert _angle_deg(ref_base.xmat.reshape(3, 3), twin_base.xmat.reshape(3, 3)) < 1e-6
    assert np.linalg.norm(twin_base.xpos - ref_base.xpos) < 0.025


def _poses():
    rng = np.random.default_rng(0)
    yield "home", scene.HOME_ARM_Q
    yield "zero", {s: np.zeros(6) for s in scene.SIDES}
    for i in range(3):
        yield f"random{i}", {s: rng.uniform(-np.pi, np.pi, 6) for s in scene.SIDES}


def _in_frame(frame, pos, rot):
    frame_rot = frame.xmat.reshape(3, 3)
    return frame_rot.T @ (pos - frame.xpos), frame_rot.T @ rot


@pytest.mark.parametrize("label, q", list(_poses()))
@pytest.mark.parametrize("side", scene.SIDES)
def test_flange_matches_reference_relative_to_base(reference, twin, side, label, q):
    ref_model, ref_data = reference
    twin_model, twin_data = twin
    _set_arms(ref_model, ref_data, q, lambda s, n: REFERENCE_PREFIX[s] + n)
    _set_arms(twin_model, twin_data, q, lambda s, n: f"{s}/{n}")

    wrist3 = ref_data.body(REFERENCE_PREFIX[side] + "wrist_3_link")
    wrist3_rot = wrist3.xmat.reshape(3, 3)
    ref_pos, ref_rot = _in_frame(
        ref_data.body(REFERENCE_PREFIX[side][:-1] + "_"),
        wrist3.xpos + wrist3_rot @ REFERENCE_FLANGE_OFFSET,
        wrist3_rot @ TOOL0_IN_REFERENCE_WRIST3.as_matrix(),
    )
    tool0 = twin_data.body(f"{side}/tool0")
    twin_pos, twin_rot = _in_frame(twin_data.body(f"{side}/base"), tool0.xpos, tool0.xmat.reshape(3, 3))

    assert np.linalg.norm(twin_pos - ref_pos) < FLANGE_POS_TOL
    assert _angle_deg(ref_rot, twin_rot) < ANGLE_TOL_DEG
