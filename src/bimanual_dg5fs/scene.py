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

# Object: one per run, chosen by name. Every one is the body "object" with the
# free joint "object_free"; an articulated Object adds the joints in OBJECT_JOINTS,
# and a Scene with more than one free body adds the bodies in EXTRA_OBJECTS, each
# with the free joint "<name>_free".
OBJECTS = ("cube", "basin", "book", "basin_cube", "bag", "bag_sealed")

# cube: 15 cm cube resting on the Table between the two palms at Home.
OBJECT_SIZE = 0.15                     # edge length [m]
OBJECT_MASS = 0.2                      # [kg]
OBJECT_XY = (0.0, -0.60)

# basin: a round plastic tub with one loop handle towards each Hand (on +-x),
# to be lifted by both handles. It is built from primitives, not a mesh: MuJoCo
# collides a mesh as its convex hull, which would fill the tub.
BASIN_XY = (0.0, -0.58)
BASIN_TOP_RADIUS = 0.19                # outer radius at the rim [m]
BASIN_BOTTOM_RADIUS = 0.15             # outer radius at the bottom [m]
BASIN_HEIGHT = 0.14
BASIN_WALL = 0.006                     # wall and bottom thickness
BASIN_SEGMENTS = 24                    # flat wall panels around the circumference
BASIN_LIP_RADIUS = 0.008               # rounded rim
BASIN_HANDLE_LENGTH = 0.13             # grip bar, along y
BASIN_HANDLE_RADIUS = 0.01
BASIN_HANDLE_GAP = 0.04                # clearance between rim and grip bar, for the fingers
BASIN_MASS = 0.5                       # [kg]

# book: a closed hardcover lying on the Table, front cover up, to be opened.
# The body "object" is the back cover, the page block and the spine; the front
# cover is the child body "book_cover" on the hinge "book_cover_hinge" along the
# spine (0 = closed, opening lifts the fore-edge). The spine is on +x, towards
# the left Hand, so the cover opens over the spine towards it, as a book read
# from the robot's side would. The covers overhang the pages by BOOK_SQUARE at
# head, tail and fore-edge: the lip a fingertip lifts the cover by.
BOOK_XY = (0.0, -0.58)
BOOK_WIDTH = 0.17                      # spine to fore-edge, along x [m]
BOOK_LENGTH = 0.24                     # head to tail, along y
BOOK_PAGES_THICKNESS = 0.03
BOOK_BOARD = 0.003                     # cover and spine board thickness
BOOK_SQUARE = 0.008                    # cover overhang beyond the pages
BOOK_PAGES_MASS = 0.55                 # [kg]
BOOK_COVER_MASS = 0.06                 # each cover; the spine weighs a fifth of it
BOOK_HINGE_RANGE_DEG = (0.0, 200.0)    # past 180 deg the open cover rests on the Table
BOOK_HINGE_DAMPING = 0.01              # [N m s/rad]
BOOK_HINGE_FRICTION = 0.005            # [N m]: an open cover stays where it is left
# Sliding, torsional, rolling friction of the book's geoms (MuJoCo's default is
# 1, 0.005, 0.0001). A contact takes the larger of its two geoms' values, so this
# grips the fingertips more firmly; it also holds the book on the Table better.
BOOK_FRICTION = (1.5, 0.01, 0.0001)
# bag: an industrial bag of coffee lying flat where the basin stands, its long side
# between the Hands, to be lifted by its short ends. A soft body cannot reach the
# training stack (Newton hands MuJoCo-Warp rigid bodies and joints only), so it is
# BAG_SEGMENTS rigid boxes along its length; at each junction a bending hinge (about
# y) and a stiffer twisting one (about x), both sprung and damped. The bending
# stiffness is set so that lifted by its two end segments, held level as two Hands
# would (free to come closer), the bag sags by BAG_SAG at its centre; soft contacts
# (BAG_SOLREF) let a squeezing fingertip sink in by about BAG_SQUEEZE; a cylinder on
# each joint axis fills the wedge bending
# opens between two boxes, and each short end (where the Hands grasp) is rounded by a
# half elliptic cylinder (a convex mesh). The plain bag closes like a real sack: the underside of
# its end segments ramps up by BAG_END_LIFT towards the end (a convex mesh too), leaving the room a
# finger slides under; bag_sealed keeps square ends behind its seams. The body "object" is the segment left of the centre,
# with its frame at the centre of the bag's underside. bag_sealed adds a flat
# sealed seam (BAG_SEAM, rigid) beyond each short end, for the fingers to close on.
BAG_XY = BASIN_XY
BAG_SIZE = (0.35, 0.22, 0.10)          # length (x, between the Hands), width (y), height [m]
BAG_MASS = 2.5                         # [kg], seams included
BAG_SEGMENTS = 6                       # enough for the sag to read as a curve, not a kink
BAG_SEAM = (0.04, 0.22, 0.01)          # beyond each short end: length, width, thickness [m]
BAG_SEAM_MASS = 0.03                   # each
BAG_END_ROUND = 0.025                  # each short end is rounded: a half elliptic cylinder as tall as the bag, sticking out this far [m]
BAG_END_LIFT = 0.025                   # the plain bag's end segments: their underside ramps up to this far off the Table at the end [m]
BAG_SAG = 0.04                         # centre sag lifted by its end segments, held level [m] (calibration target)
BAG_SQUEEZE = 0.005                    # fingertip sinking at about 10 N [m] (calibration target)
BAG_BEND_STIFFNESS = 1.42              # [N m/rad] per junction, calibrated to BAG_SAG (bends up to 9 deg)
BAG_TWIST_STIFFNESS = 3.0 * BAG_BEND_STIFFNESS
BAG_BEND_DAMPING = 0.05                # [N m s/rad], about critical for a segment
BAG_TWIST_DAMPING = 3.0 * BAG_BEND_DAMPING
BAG_BEND_RANGE_DEG = (-60.0, 60.0)
BAG_TWIST_RANGE_DEG = (-30.0, 30.0)
BAG_SOLREF = (0.143, 1.0)              # contact time constant [s] and damping ratio, calibrated to BAG_SQUEEZE
BAG_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
_BAG_JUNCTIONS = range(BAG_SEGMENTS - 1)
OBJECT_JOINTS = {"cube": (), "basin": (), "book": ("book_cover_hinge",), "basin_cube": (),
                 **{name: tuple(f"bag_{kind}_{j}" for j in _BAG_JUNCTIONS for kind in ("bend", "twist"))
                    for name in ("bag", "bag_sealed")}}

# basin_cube: the basin where the basin-only Scene has it (the Object), and the
# 15 cm cube (the extra body "cube") on the Table between it and the Arm bases,
# BASIN_CUBE_GAP from the rim: for putting the cube in the basin, then lifting both.
BASIN_CUBE_GAP = 0.05
BASIN_CUBE_XY = (BASIN_XY[0], BASIN_XY[1] + BASIN_TOP_RADIUS + BASIN_CUBE_GAP + OBJECT_SIZE / 2.0)
EXTRA_OBJECTS = {"basin_cube": ("cube",)}

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


def _add_world(spec, object_name):
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

    if object_name not in OBJECTS:
        raise ValueError(f"unknown Object {object_name!r}; choose one of {OBJECTS}")
    obj = spec.worldbody.add_body(name="object")
    obj.add_freejoint(name="object_free")
    if object_name == "cube":
        obj.pos = (OBJECT_XY[0], OBJECT_XY[1], TABLE_TOP_Z + OBJECT_SIZE / 2.0)
        obj.add_geom(name="object", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=(OBJECT_SIZE / 2.0,) * 3, mass=OBJECT_MASS,
                     rgba=(0.85, 0.30, 0.25, 1.0))
    elif object_name in ("basin", "basin_cube"):
        obj.pos = (BASIN_XY[0], BASIN_XY[1], TABLE_TOP_Z)
        _add_basin_geoms(obj)
        if object_name == "basin_cube":
            cube = spec.worldbody.add_body(name="cube")
            cube.add_freejoint(name="cube_free")
            cube.pos = (BASIN_CUBE_XY[0], BASIN_CUBE_XY[1], TABLE_TOP_Z + OBJECT_SIZE / 2.0)
            cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=(OBJECT_SIZE / 2.0,) * 3,
                          mass=OBJECT_MASS, rgba=(0.85, 0.30, 0.25, 1.0))
    elif object_name == "book":
        obj.pos = (BOOK_XY[0], BOOK_XY[1], TABLE_TOP_Z)
        _add_book(obj)
    else:
        obj.pos = (BAG_XY[0], BAG_XY[1], TABLE_TOP_Z)
        _add_bag(spec, obj, sealed=object_name == "bag_sealed")
        # The Table keeps its own hard contact with the soft bag, which would otherwise sink into it.
        spec.geom("table").priority = 1
        # No contacts within the bag: the joint cylinders beside the root segment overlap, and a
        # bag this stiff never folds onto itself (the training stack has no self-contacts either).
        bodies = ["object"] + [f"bag_segment_{i}" for i in range(BAG_SEGMENTS) if i != BAG_SEGMENTS // 2 - 1]
        for i, a in enumerate(bodies):
            for b in bodies[i + 1:]:
                spec.add_exclude(name=f"{a}__{b}", bodyname1=a, bodyname2=b)


def _add_bag(spec, body, sealed):
    """The bag in the frame of its centre-left segment ("object"): origin at the centre of the
    bag's underside. Segments i = 0 .. BAG_SEGMENTS-1 run along +x; the chain branches from
    the root towards both ends."""
    box = mujoco.mjtGeom.mjGEOM_BOX
    kraft, seam_rgba = (0.60, 0.46, 0.30, 1.0), (0.45, 0.34, 0.22, 1.0)
    length, width, height = BAG_SIZE
    segment = length / BAG_SEGMENTS
    mass = (BAG_MASS - (2 * BAG_SEAM_MASS if sealed else 0.0)) / BAG_SEGMENTS
    root = BAG_SEGMENTS // 2 - 1
    soft = {"solref": BAG_SOLREF, "solimp": BAG_SOLIMP}

    lift = 0.0 if sealed else BAG_END_LIFT          # the end segments' underside rise at the end
    ends_at = (0, BAG_SEGMENTS - 1)

    def add_segment(parent, i, pos, sign):
        # The body sits on the junction with its parent (mid-height); its box extends towards ``sign``.
        child = parent.add_body(name=f"bag_segment_{i}", pos=pos)
        junction = i if sign < 0 else i - 1                  # junction j lies between segments j and j+1
        child.add_joint(name=f"bag_bend_{junction}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0.0, 1.0, 0.0),
                        stiffness=BAG_BEND_STIFFNESS, damping=BAG_BEND_DAMPING, range=BAG_BEND_RANGE_DEG,
                        armature=1e-3)
        child.add_joint(name=f"bag_twist_{junction}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(1.0, 0.0, 0.0),
                        stiffness=BAG_TWIST_STIFFNESS, damping=BAG_TWIST_DAMPING, range=BAG_TWIST_RANGE_DEG,
                        armature=1e-3)
        if i in ends_at and lift > 0.0:
            # An end segment of the plain bag: a prism whose underside ramps up towards the end.
            outline = [(0.0, -height / 2.0), (0.0, height / 2.0), (sign * segment, height / 2.0),
                       (sign * segment, -height / 2.0 + lift)]
            spec.add_mesh(name=f"bag_{i}", uservert=np.ravel([(x, y, z) for x, z in outline
                                                              for y in (-width / 2.0, width / 2.0)]).tolist())
            child.add_geom(name=f"bag_{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"bag_{i}", mass=mass,
                           rgba=kraft, **soft)
        else:
            child.add_geom(name=f"bag_{i}", type=box, size=(segment / 2.0, width / 2.0, height / 2.0),
                           pos=(sign * segment / 2.0, 0.0, 0.0), mass=mass, rgba=kraft, **soft)
        # A massless cylinder on the joint axis, as tall as the bag: it turns about its own
        # axis, so it fills the wedge the bending opens between two boxes at any angle.
        child.add_geom(name=f"bag_joint_{junction}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=(height / 2.0, width / 2.0, 0.0), quat=(np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0),
                       mass=0.0, rgba=kraft, **soft)
        return child

    # Root segment: the one just left of the centre, its frame at the bag's centre.
    body.add_geom(name=f"bag_{root}", type=box, size=(segment / 2.0, width / 2.0, height / 2.0),
                  pos=(-segment / 2.0, 0.0, height / 2.0), mass=mass, rgba=kraft, **soft)
    ends = {}
    for sign, indices in ((-1, range(root - 1, -1, -1)), (+1, range(root + 1, BAG_SEGMENTS))):
        parent = body
        for k, i in enumerate(indices):
            # The first junction is on the root's edge towards ``sign``; the next ones one segment further.
            first = (-segment if sign < 0 else 0.0, 0.0, height / 2.0)
            parent = add_segment(parent, i, first if k == 0 else (sign * segment, 0.0, 0.0), sign)
        ends[sign] = parent
    # Rounded short ends: half an elliptic cylinder (semi-axes BAG_END_ROUND along x and half the
    # end face's height along z, extruded across the width) on each end segment's outer face.
    # Massless, like the joint fillers, so the calibration is unchanged.
    angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, 13)
    face_half, face_middle = (height - lift) / 2.0, lift / 2.0
    for sign, end in ends.items():
        side = "left" if sign > 0 else "right"
        outline = [(sign * (segment + BAG_END_ROUND * np.cos(a)), face_middle + face_half * np.sin(a)) for a in angles]
        vertices = [(x, y, z) for x, z in outline for y in (-width / 2.0, width / 2.0)]
        spec.add_mesh(name=f"bag_end_{side}", uservert=np.ravel(vertices).tolist())
        end.add_geom(name=f"bag_end_{side}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"bag_end_{side}",
                     mass=0.0, rgba=kraft, **soft)
    if sealed:
        seam_length, seam_width, seam_thickness = BAG_SEAM
        for sign, end in ends.items():
            end.add_geom(name=f"bag_seam_{'left' if sign > 0 else 'right'}", type=box,
                         size=(seam_length / 2.0, seam_width / 2.0, seam_thickness / 2.0),
                         pos=(sign * (segment + BAG_END_ROUND + seam_length / 2.0), 0.0, 0.0), mass=BAG_SEAM_MASS,
                         rgba=seam_rgba, **soft)


def _add_book(body):
    """The book in its body frame: origin at the centre of the back cover's underside."""
    box = mujoco.mjtGeom.mjGEOM_BOX
    cloth, paper = (0.15, 0.32, 0.25, 1.0), (0.93, 0.90, 0.80, 1.0)
    friction = BOOK_FRICTION
    w, length, b, p, sq = BOOK_WIDTH, BOOK_LENGTH, BOOK_BOARD, BOOK_PAGES_THICKNESS, BOOK_SQUARE
    body.add_geom(name="book_back_cover", type=box, size=(w / 2.0, length / 2.0, b / 2.0),
                  pos=(0.0, 0.0, b / 2.0), mass=BOOK_COVER_MASS, rgba=cloth, friction=friction)
    body.add_geom(name="book_spine", type=box, size=(b / 2.0, length / 2.0, p / 2.0 + b),
                  pos=(w / 2.0 - b / 2.0, 0.0, p / 2.0 + b), mass=BOOK_COVER_MASS / 5.0, rgba=cloth,
                  friction=friction)
    # Pages: from the fore-edge square to the spine, between the covers.
    body.add_geom(name="book_pages", type=box, size=((w - sq - b) / 2.0, length / 2.0 - sq, p / 2.0),
                  pos=((sq - b) / 2.0, 0.0, b + p / 2.0), mass=BOOK_PAGES_MASS, rgba=paper, friction=friction)
    # Front cover, hinged at the top of the spine. It is the child of the page
    # block's body, so MuJoCo never collides the two: the joint limit closes it.
    cover = body.add_body(name="book_cover", pos=(w / 2.0 - b / 2.0, 0.0, b + p + b / 2.0))
    cover.add_joint(name="book_cover_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0.0, 1.0, 0.0),
                    range=BOOK_HINGE_RANGE_DEG, damping=BOOK_HINGE_DAMPING,          # degrees (MjSpec compiler)
                    frictionloss=BOOK_HINGE_FRICTION, armature=1e-4)
    cover.add_geom(name="book_front_cover", type=box, size=((w - b) / 2.0, length / 2.0, b / 2.0),
                   pos=(-w / 2.0, 0.0, 0.0), mass=BOOK_COVER_MASS, rgba=cloth, friction=friction)


def _add_basin_geoms(body):
    """The basin in its body frame: origin at the centre of its bottom, on the Table."""
    rgba = (0.35, 0.60, 0.85, 1.0)
    geoms = []                         # (geom, volume), to spread BASIN_MASS uniformly

    def capsule(start, end, radius):
        g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=(radius, 0.0, 0.0),
                          fromto=(*start, *end), rgba=rgba)
        length = np.linalg.norm(np.subtract(end, start))
        geoms.append((g, np.pi * radius**2 * length + 4.0 / 3.0 * np.pi * radius**3))

    # Wall: flat panels tilted outwards; their outer bottom edges rest on the Table.
    flare = BASIN_TOP_RADIUS - BASIN_BOTTOM_RADIUS
    tilt = np.arctan2(flare, BASIN_HEIGHT)
    half = (BASIN_WALL / 2.0, BASIN_TOP_RADIUS * np.sin(np.pi / BASIN_SEGMENTS),
            np.hypot(BASIN_HEIGHT, flare) / 2.0)
    for i in range(BASIN_SEGMENTS):
        angle = 2.0 * np.pi * i / BASIN_SEGMENTS
        radius = BASIN_BOTTOM_RADIUS + flare / 2.0 - half[0] * np.cos(tilt)
        height = BASIN_HEIGHT / 2.0 + half[0] * np.sin(tilt)
        g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
                          pos=(radius * np.cos(angle), radius * np.sin(angle), height),
                          quat=R.from_euler("ZY", (angle, tilt)).as_quat(scalar_first=True),
                          rgba=rgba)
        geoms.append((g, 8.0 * np.prod(half)))

    # Bottom, 1 mm above the Table so that the wall carries the basin.
    bottom_radius = BASIN_BOTTOM_RADIUS - BASIN_WALL
    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=(bottom_radius, BASIN_WALL / 2.0, 0.0),
                      pos=(0.0, 0.0, 0.001 + BASIN_WALL / 2.0), rgba=rgba)
    geoms.append((g, np.pi * bottom_radius**2 * BASIN_WALL))

    # Rounded rim.
    rim = [(BASIN_TOP_RADIUS * np.cos(a), BASIN_TOP_RADIUS * np.sin(a), BASIN_HEIGHT)
           for a in np.linspace(0.0, 2.0 * np.pi, BASIN_SEGMENTS + 1)]
    for start, end in zip(rim[:-1], rim[1:]):
        capsule(start, end, BASIN_LIP_RADIUS)

    # Handles: a grip bar along y at rim height on each side, held by two arms.
    bar_x = BASIN_TOP_RADIUS + BASIN_LIP_RADIUS + BASIN_HANDLE_GAP + BASIN_HANDLE_RADIUS
    y = BASIN_HANDLE_LENGTH / 2.0
    rim_x = np.sqrt(BASIN_TOP_RADIUS**2 - y**2)
    for sign in (1.0, -1.0):
        capsule((sign * bar_x, -y, BASIN_HEIGHT), (sign * bar_x, y, BASIN_HEIGHT), BASIN_HANDLE_RADIUS)
        for end_y in (-y, y):
            capsule((sign * rim_x, end_y, BASIN_HEIGHT), (sign * bar_x, end_y, BASIN_HEIGHT),
                    BASIN_HANDLE_RADIUS)

    density = BASIN_MASS / sum(volume for _, volume in geoms)
    for g, _ in geoms:
        g.density = density


def basin_model():
    """The basin alone, compiled: body 1 carries its geoms, mass and inertia as in the Scene.

    For simulators other than MuJoCo (the training stack), which rebuild the basin
    from these geoms instead of repeating its dimensions."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="object_free")
    _add_basin_geoms(body)
    return spec.compile()


def book_model():
    """The book alone, compiled: body 1 ("object", free) carries the back cover, spine and
    pages, body 2 ("book_cover") the front cover on the hinge "book_cover_hinge", with the
    Scene's geoms, masses, inertias, friction and hinge range, damping, friction and armature.

    For simulators other than MuJoCo (the training stack), which rebuild the book from
    this model instead of repeating its dimensions."""
    spec = mujoco.MjSpec()
    spec.compiler.autolimits = True
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="object_free")
    _add_book(body)
    return spec.compile()


def bag_model(sealed=False):
    """The bag alone, compiled (like book_model): body "object" (free) and the segment chain."""
    spec = mujoco.MjSpec()
    spec.compiler.autolimits = True
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="object_free")
    _add_bag(spec, body, sealed)
    return spec.compile()


def object_joint_ranges(object_name):
    """``{joint: (lower, upper)}`` [rad] of an Object's joints (OBJECT_JOINTS); empty for a rigid one."""
    names = OBJECT_JOINTS[object_name]
    if not names:
        return {}
    model = {"book": book_model, "bag": bag_model, "bag_sealed": lambda: bag_model(sealed=True)}[object_name]()
    return {name: tuple(float(v) for v in model.joint(name).range) for name in names}


def object_metadata(object_name):
    """What a Demonstration records about its Object."""
    if object_name == "cube":
        return {"object": "cube", "object_size_m": OBJECT_SIZE, "object_mass_kg": OBJECT_MASS}
    if object_name == "book":
        return {"object": "book", "object_mass_kg": BOOK_PAGES_MASS + 2.2 * BOOK_COVER_MASS,
                "object_joints": list(OBJECT_JOINTS["book"]),
                "book": {"width_m": BOOK_WIDTH, "length_m": BOOK_LENGTH,
                         "pages_thickness_m": BOOK_PAGES_THICKNESS, "board_m": BOOK_BOARD,
                         "square_m": BOOK_SQUARE, "hinge_range_deg": list(BOOK_HINGE_RANGE_DEG)}}
    if object_name in ("bag", "bag_sealed"):
        return {"object": object_name, "object_mass_kg": BAG_MASS,
                "object_joints": list(OBJECT_JOINTS[object_name]),
                "bag": {"size_m": list(BAG_SIZE), "segments": BAG_SEGMENTS,
                        "seam_m": list(BAG_SEAM) if object_name == "bag_sealed" else None,
                        "bend_stiffness_nm_per_rad": BAG_BEND_STIFFNESS,
                        "twist_stiffness_nm_per_rad": BAG_TWIST_STIFFNESS,
                        "bend_range_deg": list(BAG_BEND_RANGE_DEG), "twist_range_deg": list(BAG_TWIST_RANGE_DEG),
                        "contact_solref": list(BAG_SOLREF)}}
    basin = {"object": object_name, "object_mass_kg": BASIN_MASS,
             "basin": {"top_radius_m": BASIN_TOP_RADIUS, "bottom_radius_m": BASIN_BOTTOM_RADIUS,
                       "height_m": BASIN_HEIGHT, "wall_m": BASIN_WALL,
                       "handle_length_m": BASIN_HANDLE_LENGTH, "handle_radius_m": BASIN_HANDLE_RADIUS,
                       "handle_gap_m": BASIN_HANDLE_GAP}}
    if object_name == "basin_cube":
        basin["extra_objects"] = {"cube": {"size_m": OBJECT_SIZE, "mass_kg": OBJECT_MASS,
                                           "start_xy": list(BASIN_CUBE_XY)}}
    return basin


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


def build_spec(object_name="cube"):
    spec = mujoco.MjSpec()
    spec.modelname = "bimanual_dg5fs_scene"
    spec.compiler.autolimits = True
    _add_world(spec, object_name)
    _add_limbs(spec)
    _add_actuators(spec)
    _add_resting_contact_excludes(spec)
    model = spec.compile()
    qpos = home_qpos(model)
    spec.add_key(name="home", qpos=qpos, ctrl=_home_ctrl(model, qpos))
    return spec


def build_model(object_name="cube"):
    return build_spec(object_name).compile()


def reset_to_home(model, data):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
