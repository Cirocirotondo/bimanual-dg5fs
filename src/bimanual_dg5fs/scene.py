"""The Scene: both Limbs at their Base poses, the Table and the Object, in MuJoCo.

The Limb URDFs in ``assets/`` are the source of truth (docs/adr/0001); this
module only places them and adds what MuJoCo needs on top (actuators, contact
excludes, the ``home`` keyframe). Nothing here changes the robot itself.
"""
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
SIDES = ("left", "right")


def limb_urdf(side):
    return ASSETS_DIR / f"ur5e_{side}_dg5fs.urdf"


# -----------------------------------------------------------------------------
# Sides are named from the operator's position in this Twin, which is opposite
# to the Reference setup's: our right Limb is the Reference ur5_1 ("left" in
# Nando's code, x = -0.66) and our left Limb is the Reference ur5_2.
REFERENCE_ARM = {"right": "ur5_1", "left": "ur5_2"}

# Base poses: the Reference setup's (bimanual_scene_cube.xml, bodies ur5_1_ at
# (-0.66, 0, 0.04) and ur5_2_ at (0.66, -0.005, 0.02)), made mirror-symmetric
# about the plane x = 0 and set down on the Table top (z = 0.02, the Reference
# calibration's floor height). They are poses of the UR controller's base frame,
# which is the URDF link `base`, not `base_link`: the URDF rotates base_link ->
# base by pi about z, so each Limb is attached with that extra yaw. Joint angles
# therefore mean the same thing here as on the real Arms.
# -----------------------------------------------------------------------------
BASE_POS = {
    "right": (-0.66, 0.0, 0.02),
    "left": (0.66, 0.0, 0.02),
}
BASE_YAW_DEG = {"right": -90.0, "left": 90.0}
URDF_BASE_LINK_TO_BASE_YAW_DEG = 180.0

ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
HAND_JOINTS = tuple(f"joint_{f}_{j}" for f in range(1, 6) for j in range(1, 5))

# With the bases above, the left Limb is the mirror image of the right one
# across x = 0 (Hands included) when q_left = MIRROR_OFFSET - q_right.
MIRROR_OFFSET = (2 * np.pi, -np.pi, 0.0, np.pi, 0.0, np.pi)


def mirror_arm_q(q):
    """Arm joint angles of the mirror image, across x = 0, of the other Limb."""
    return tuple(c - v for c, v in zip(MIRROR_OFFSET, q))


# Home pose: the right Limb takes the Reference teleop home HOME_POSES[0.55]
# (bimanual_singlehand.py, ur5_1), with wrist_2 turned by 90 deg so the Hand
# points forward (-y, away from the line between the bases) instead of at the
# other Hand, and wrist_3 turned by 180 deg so the thumb points up. The left
# Limb is its mirror image. Hands fully open (all Hand joints at 0).
REFERENCE_HOME_ARM_Q_RIGHT = (3.4696, -1.0225, 1.5602, -0.4681, 0.3556, 3.0722)
HOME_TURN_RIGHT = {"wrist_2_joint": np.pi / 2, "wrist_3_joint": -np.pi}
_home_right = tuple(q + HOME_TURN_RIGHT.get(name, 0.0)
                    for name, q in zip(ARM_JOINTS, REFERENCE_HOME_ARM_Q_RIGHT))
HOME_ARM_Q = {"right": _home_right, "left": mirror_arm_q(_home_right)}

# -----------------------------------------------------------------------------
# PLACEHOLDERS until measured on the real rig.
# -----------------------------------------------------------------------------
# Table: a box whose top is flush with the Arm base plane (the Reference base
# calibration puts both bases on one surface at z = 0.02). It reaches 0.4 m
# behind the bases and 1.0 m in front of them, on the side the Hands point to.
TABLE_TOP_Z = 0.02
TABLE_SIZE = (1.6, 1.4, 0.75)          # x (along the Arms), y, height [m]
TABLE_CENTER_XY = (0.0, -0.3)

# Object: 15 cm cube resting on the Table between the two palms at Home.
OBJECT_SIZE = 0.15                     # edge length [m]
OBJECT_MASS = 0.2                      # [kg]
OBJECT_XY = (0.0, -0.60)

# Position-actuator gains. Arm: the Reference setup's UR5e MJCF gains.
# Hand: conservative values, not yet tuned against the real DG-5F-S.
ARM_KP = (2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0)
ARM_KV = (400.0, 400.0, 400.0, 100.0, 100.0, 100.0)
ARM_FORCE = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
ARM_ARMATURE = 0.1
HAND_KP = 5.0
HAND_KV = 0.1
HAND_FORCE = 7.5                       # URDF effort limit
HAND_ARMATURE = 0.005


def _limb_spec(side):
    """Load one Limb URDF as an MjSpec, keeping every link as its own body."""
    text = limb_urdf(side).read_text()
    # MuJoCo cannot decode the .dae visuals, so it shows the collision meshes.
    mj_tag = (f'<mujoco><compiler meshdir="{ASSETS_DIR}" strippath="false" '
              f'fusestatic="false" discardvisual="true"/></mujoco>')
    text = text.replace(f'<robot name="ur5e_{side}_dg5fs">',
                        f'<robot name="ur5e_{side}_dg5fs">\n  {mj_tag}', 1)
    return mujoco.MjSpec.from_string(text)


def _add_world(spec):
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0

    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    light = spec.worldbody.add_light()
    light.pos = (0.0, 0.0, 3.0)
    light.dir = (0.0, 0.0, -1.0)
    light.diffuse = (0.8, 0.8, 0.8)

    floor_z = TABLE_TOP_Z - TABLE_SIZE[2]
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=(3.0, 3.0, 0.05), pos=(0.0, 0.0, floor_z),
                            rgba=(0.20, 0.25, 0.28, 1.0))

    table = spec.worldbody.add_body(name="table")
    half = np.asarray(TABLE_SIZE) / 2.0
    table.pos = (TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_TOP_Z - half[2])
    table.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
                   rgba=(0.55, 0.45, 0.35, 1.0))

    obj = spec.worldbody.add_body(name="object")
    obj.pos = (OBJECT_XY[0], OBJECT_XY[1], TABLE_TOP_Z + OBJECT_SIZE / 2.0)
    obj.add_freejoint(name="object_free")
    obj.add_geom(name="object", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=(OBJECT_SIZE / 2.0,) * 3, mass=OBJECT_MASS,
                 rgba=(0.85, 0.30, 0.25, 1.0))


def _add_limbs(spec):
    for side in SIDES:
        frame = spec.worldbody.add_frame(name=f"{side}_base_frame")
        frame.pos = BASE_POS[side]
        yaw = BASE_YAW_DEG[side] - URDF_BASE_LINK_TO_BASE_YAW_DEG
        frame.quat = R.from_euler("z", yaw, degrees=True).as_quat(scalar_first=True)
        limb = _limb_spec(side)
        frame.attach_body(limb.worldbody.first_body(), f"{side}/", "")
    # The real UR and Tesollo controllers compensate gravity, and so does the
    # training stack (Newton authors gravcomp = 1 on every link).
    for body in spec.bodies:
        if body.name.split("/", 1)[0] in SIDES:
            body.gravcomp = 1.0


def _add_actuators(spec):
    for side in SIDES:
        for i, name in enumerate(ARM_JOINTS):
            _position_actuator(spec, f"{side}/{name}", ARM_KP[i], ARM_KV[i], ARM_FORCE[i], ARM_ARMATURE)
        for name in HAND_JOINTS:
            _position_actuator(spec, f"{side}/{name}", HAND_KP, HAND_KV, HAND_FORCE, HAND_ARMATURE)


def _position_actuator(spec, joint_name, kp, kv, force, armature):
    joint = spec.joint(joint_name)
    joint.armature = armature
    act = spec.add_actuator(name=joint_name, target=joint_name,
                            trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.gainprm[0] = kp
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.biasprm[0:3] = (0.0, -kp, -kv)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = joint.range
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = (-force, force)


def home_qpos(model):
    """Full qpos for the Home pose; the Object rests at its start pose."""
    qpos = model.qpos0.copy()
    for side in SIDES:
        for name, q in zip(ARM_JOINTS, HOME_ARM_Q[side]):
            qpos[model.joint(f"{side}/{name}").qposadr[0]] = q
        for name in HAND_JOINTS:
            qpos[model.joint(f"{side}/{name}").qposadr[0]] = 0.0
    return qpos


def _home_ctrl(model, qpos):
    return np.array([qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]] for a in range(model.nu)])


def _limb_of(model, body_id):
    name = model.body(body_id).name
    side = name.split("/", 1)[0]
    return side if side in SIDES else None


def _add_resting_contact_excludes(spec):
    """Exclude contacts between bodies of the same Limb that already touch at
    rest (rigidly mounted neighbours such as the Mount and the wrist), so they
    do not fight each other. Contacts between Limbs, and with Table/Object, stay on."""
    model = spec.compile()
    data = mujoco.MjData(model)
    pairs = set()
    for qpos in (model.qpos0, home_qpos(model)):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for c in data.contact[: data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            side = _limb_of(model, b1)
            if side is not None and side == _limb_of(model, b2):
                pairs.add(tuple(sorted((model.body(b1).name, model.body(b2).name))))
    for b1, b2 in sorted(pairs):
        spec.add_exclude(name=f"{b1}__{b2}", bodyname1=b1, bodyname2=b2)
    return sorted(pairs)


def build_spec():
    spec = mujoco.MjSpec()
    spec.modelname = "bimanual_dg5fs_scene"
    spec.compiler.autolimits = True
    _add_world(spec)
    _add_limbs(spec)
    _add_actuators(spec)
    _add_resting_contact_excludes(spec)
    model = spec.compile()
    qpos = home_qpos(model)
    spec.add_key(name="home", qpos=qpos, ctrl=_home_ctrl(model, qpos))
    return spec


def build_model():
    return build_spec().compile()


def reset_to_home(model, data):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
