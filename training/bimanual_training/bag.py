"""The coffee bag Objects "bag" and "bag_sealed" (CONTEXT.md) as an Isaac Lab articulation.

The bag stands in for a soft body: scene.BAG_SEGMENTS rigid boxes along its length,
joined at each junction by a sprung bending hinge (about y) and a sprung twisting hinge
(about x). It is authored here as USD, read from the Scene's compiled MuJoCo bag
(bimanual_dg5fs/scene.py ``bag_model``), so its dimensions and calibration live in
scene.py only:

* the root body "object" (the segment left of the centre, frame at the centre of the
  bag's underside) is the articulation root (floating: Newton gives it a free joint);
  every other segment is a rigid body prim under it, at its pose with the joints at 0;
* at each junction two PhysicsRevoluteJoints between the same two bodies, the bend
  first: Newton merges them into one D6 joint whose angular DOFs keep this order, which
  SolverMuJoCo writes back as two hinges in the child body, as in the Scene;
* passive: no drive. MuJoCo's spring (``mjc:stiffness``, ``mjc:springref``), Newton's
  damping, armature and friction, the limit's ``mjc:solreflimit`` / ``mjc:solimplimit``;
* every collider (boxes, the joint cylinders, the convex meshes of the rounded ends, the seams)
  carries the Scene's soft contact (``mjc:solref``, ``mjc:solimp``), its
  friction, and the massless joint cylinders and rounded ends add no mass: each body's mass and inertia
  are authored from the model;
* self-collision is off. MuJoCo never collides a body with its parent; the only other
  bag-bag contact in the Scene, between the two joint cylinders beside the root, acts
  along the line through both hinge axes and so exerts no torque on any joint.

The Table must keep its hard contact with the bag (the Scene gives it ``priority = 1``):
``harden_against_bag`` authors that on a collider prim.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import mujoco
import numpy as np
import torch
from pxr import Gf, Sdf, UsdPhysics, Vt

from isaaclab.sim import schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass

from bimanual_dg5fs import scene
from bimanual_training.basin import _xyzw
from bimanual_training.book import _friction, _mass, _quatf

BAGS = ("bag", "bag_sealed")
ROOT_BODY = 1                          # "object" in scene.bag_model()


def bag_model(object_name):
    if object_name not in BAGS:
        raise ValueError(f"{object_name!r} is not a bag; choose one of {BAGS}")
    return scene.bag_model(sealed=object_name == "bag_sealed")


def _body_poses(model):
    """``{body: (pos, quat wxyz)}`` of every bag body in the root body's frame, joints at 0."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[0:7] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_kinematics(model, data)
    return {b: (data.xpos[b].copy(), data.xquat[b].copy()) for b in range(1, model.nbody)}


def end_points(object_name):
    """The bag's two short ends, the points the Hands lift it by: ``[(body name, (3,) point in that
    body's frame)]``, the +x end first (towards the left Hand, limbs.SIDES order), then the -x end.
    The outer tip of a rounded end (bag), or the centre of its sealed seam (bag_sealed)."""
    model = bag_model(object_name)
    last = scene.BAG_SEGMENTS - 1
    segment = scene.BAG_SIZE[0] / scene.BAG_SEGMENTS
    points = []
    for index, sign, side in ((last, +1.0, "left"), (0, -1.0, "right")):
        body = f"bag_segment_{index}"
        if object_name == "bag_sealed":
            geom = model.geom(f"bag_seam_{side}").id
            point = model.geom_pos[geom].copy()
        else:
            # The end segment's body sits on its inner junction (mid-height): its box reaches sign * segment,
            # the rounded end (geom bag_end_<side>) BAG_END_ROUND further, at mid-height.
            geom = model.geom(f"bag_end_{side}").id
            point = np.array([sign * (segment + scene.BAG_END_ROUND), 0.0, 0.0])
        if model.body(model.geom_bodyid[geom]).name != body:
            raise ValueError(f"{model.geom(geom).name} is expected on {body}")
        points.append((body, torch.tensor(point, dtype=torch.float32)))
    return points


def backend_joint_names(object_name):
    """The bag's joints (scene.OBJECT_JOINTS order) as the Isaac Lab articulation names them: Newton
    merges the two hinges at junction j into one D6 joint named after the first, the bend, whose
    DOFs are "bag_bend_<j>:0" (the bend) and "bag_bend_<j>:1" (the twist)."""
    names = []
    for name in scene.OBJECT_JOINTS[object_name]:
        kind, junction = name.rsplit("_", 2)[1:]
        names.append(f"bag_bend_{junction}:{0 if kind == 'bend' else 1}")
    return names


def _set(prim, name, type_name, value):
    prim.CreateAttribute(name, type_name).Set(value)


@clone
def spawn_bag(prim_path, cfg: BagCfg, translation=None, orientation=None, **kwargs):
    """``{prim_path}``: the root rigid body (articulation root) at the bag's body frame;
    ``{prim_path}/bag_segment_<i>``: the other segments; ``{prim_path}/.../geometry/geom_NN``: the
    colliders (MuJoCo geom order); ``{prim_path}/bag_bend_<j>``, ``bag_twist_<j>``: the joints."""
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    model = bag_model(cfg.object_name)
    root = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)

    sliding, torsional, rolling = _friction(model)
    physics_material = f"{prim_path}/physicsMaterial"
    spawn_physics_material(physics_material, materials.RigidBodyMaterialBaseCfg(
        static_friction=sliding, dynamic_friction=sliding, restitution=0.0), stage=stage)
    material_prim = stage.GetPrimAtPath(physics_material)
    _set(material_prim, "newton:torsionalFriction", Sdf.ValueTypeNames.Float, torsional)
    _set(material_prim, "newton:rollingFriction", Sdf.ValueTypeNames.Float, rolling)
    visual_materials = {}

    poses = _body_poses(model)
    body_paths = {ROOT_BODY: prim_path}
    for body in range(ROOT_BODY + 1, model.nbody):
        position, quat = poses[body]
        body_paths[body] = f"{prim_path}/{model.body(body).name}"
        create_prim(body_paths[body], prim_type="Xform", stage=stage,
                    translation=tuple(float(v) for v in position), orientation=_xyzw(quat))

    for g in range(model.ngeom):
        kind, size = mujoco.mjtGeom(int(model.geom_type[g])), model.geom_size[g]
        pose = {"translation": tuple(float(v) for v in model.geom_pos[g]), "orientation": _xyzw(model.geom_quat[g])}
        path = f"{body_paths[int(model.geom_bodyid[g])]}/geometry/geom_{g:02d}"
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            # A USD Cube of size 2 spans [-1, 1]: scale it by the half-extents.
            prim = create_prim(path, "Cube", stage=stage, scale=tuple(float(s) for s in size),
                               attributes={"size": 2.0}, **pose)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            # MuJoCo: radius and half-length along the geom's z; USD: radius and full length.
            prim = create_prim(path, "Cylinder", stage=stage, attributes={
                "radius": float(size[0]), "height": 2.0 * float(size[1]), "axis": "Z"}, **pose)
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            # A convex mesh (the rounded short ends): MuJoCo's vertices and hull faces, in the geom
            # frame MuJoCo moved the mesh to; collided as its convex hull, as MuJoCo does.
            mesh = int(model.geom_dataid[g])
            vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            faces = model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
            prim = create_prim(path, "Mesh", stage=stage, attributes={
                "points": Vt.Vec3fArray([Gf.Vec3f(*(float(v) for v in vertex)) for vertex in vertices]),
                "faceVertexCounts": Vt.IntArray([3] * len(faces)),
                "faceVertexIndices": Vt.IntArray([int(i) for i in np.ravel(faces)])}, **pose)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        else:
            raise ValueError(f"bag geom {g} has unsupported MuJoCo type {kind}")
        if cfg.collision_props is not None:
            schemas.define_collision_properties(path, cfg.collision_props, stage=stage)
        # The Scene's soft contact, raw (SolverMuJoCo copies an authored mjc:solref as it is).
        _set(prim, "mjc:solref", Sdf.ValueTypeNames.Double2, Gf.Vec2d(*(float(v) for v in model.geom_solref[g])))
        _set(prim, "mjc:solimp", Sdf.ValueTypeNames.DoubleArray,
             Vt.DoubleArray([float(v) for v in model.geom_solimp[g]]))
        _set(prim, "mjc:priority", Sdf.ValueTypeNames.Int, int(model.geom_priority[g]))
        color = tuple(round(float(c), 3) for c in model.geom_rgba[g][:3])
        if color not in visual_materials:
            visual_materials[color] = f"{prim_path}/visualMaterial_{len(visual_materials)}"
            visual = materials.PreviewSurfaceCfg(diffuse_color=color)
            visual.func(visual_materials[color], visual)
        bind_visual_material(path, visual_materials[color], stage=stage)
        bind_physics_material(path, physics_material, stage=stage)

    for body, path in body_paths.items():
        # The Scene's mass and inertia; the massless joint cylinders and rounded ends add nothing to them.
        _mass(stage.GetPrimAtPath(path), model, body)
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(path, cfg.rigid_props, stage=stage)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    _set(root, "newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool, False)

    # The joints, in the Scene's order: at each junction the bend (about y) before the twist (about x),
    # both at the child body's origin. Their frame is the child's frame in its parent at q = 0.
    usd_axis = {(1.0, 0.0, 0.0): "X", (0.0, 1.0, 0.0): "Y", (0.0, 0.0, 1.0): "Z"}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE or np.any(model.jnt_pos[joint_id]):
            raise ValueError("the bag's joints are expected to be hinges at their body's origin")
        name, child = model.joint(joint_id).name, int(model.jnt_bodyid[joint_id])
        parent = int(model.body_parentid[child])
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/{name}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body_paths[parent])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body_paths[child])])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in model.body_pos[child])))
        joint.CreateLocalRot0Attr(_quatf(model.body_quat[child]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateAxisAttr(usd_axis[tuple(float(v) for v in model.jnt_axis[joint_id])])
        lower, upper = (math.degrees(float(v)) for v in model.jnt_range[joint_id])
        joint.CreateLowerLimitAttr(lower)
        joint.CreateUpperLimitAttr(upper)
        dof, prim = model.jnt_dofadr[joint_id], joint.GetPrim()
        # Newton reads newton:damping per degree (USD's angle unit) and converts it to per radian;
        # friction loss [N m] and armature [kg m^2] carry no angle. mjc:stiffness [N m/rad] and
        # mjc:springref [rad] reach MuJoCo as they are.
        _set(prim, "newton:damping", Sdf.ValueTypeNames.Float, float(model.dof_damping[dof]) * math.pi / 180.0)
        _set(prim, "newton:friction", Sdf.ValueTypeNames.Float, float(model.dof_frictionloss[dof]))
        _set(prim, "newton:armature", Sdf.ValueTypeNames.Float, float(model.dof_armature[dof]))
        _set(prim, "mjc:stiffness", Sdf.ValueTypeNames.Float, float(model.jnt_stiffness[joint_id]))
        _set(prim, "mjc:springref", Sdf.ValueTypeNames.Float, float(model.qpos_spring[model.jnt_qposadr[joint_id]]))
        _set(prim, "mjc:solreflimit", Sdf.ValueTypeNames.Double2,
             Gf.Vec2d(*(float(v) for v in model.jnt_solref[joint_id])))
        _set(prim, "mjc:solimplimit", Sdf.ValueTypeNames.DoubleArray,
             Vt.DoubleArray([float(v) for v in model.jnt_solimp[joint_id]]))
    return root


def harden_against_bag(stage, collider_path):
    """Give a collider (the Table) the Scene's ``priority = 1``: its contact with the soft bag
    then takes its own (hard) solref and solimp instead of a mix with the bag's."""
    prim = stage.GetPrimAtPath(collider_path)
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError(f"{collider_path} is not a collider")
    _set(prim, "mjc:priority", Sdf.ValueTypeNames.Int, 1)


@configclass
class BagCfg(RigidObjectSpawnerCfg):
    """Spawns the Scene's bag ``object_name`` ("bag" or "bag_sealed"); masses, inertias, friction,
    contacts and joints come from the Scene, not from ``mass_props`` or a physics material."""

    func: Callable = spawn_bag
    object_name: str = "bag"
    # collision_props and rigid_props: as for the cube (object_imitation_env.py).
