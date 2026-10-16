"""The book Object (CONTEXT.md) as an Isaac Lab articulation.

The book is articulated: the free base body (back cover, spine, page block) and
the front cover on a passive hinge along the spine. It is authored here as USD,
read from the Scene's compiled MuJoCo book (bimanual_dg5fs/scene.py
``book_model``): the base is the articulation root (floating: Newton gives it a
free joint), the cover a child rigid body, the hinge a PhysicsRevoluteJoint with
the Scene's range, damping, friction loss and armature and no drive. Geoms,
masses, inertias and friction come from the model, so the dimensions live in
scene.py only. Self-collision is off, as in MuJoCo, which never collides a body
with its parent: the joint limit closes the cover onto the pages.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import mujoco
import numpy as np
from pxr import Gf, Sdf, UsdPhysics

from isaaclab.sim import schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass

from bimanual_dg5fs import scene
from bimanual_training.basin import _xyzw

BASE_BODY, COVER_BODY = 1, 2           # body ids in scene.book_model()
COVER_PRIM = "book_cover"
HINGE = scene.OBJECT_JOINTS["book"][0]


def _quatf(wxyz):
    return Gf.Quatf(*(float(v) for v in wxyz))


def _mass(prim, model, body):
    """The Scene's mass and inertia, about the MuJoCo body's inertial frame."""
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(float(model.body_mass[body]))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in model.body_ipos[body])))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(v) for v in model.body_inertia[body])))
    mass.CreatePrincipalAxesAttr(_quatf(model.body_iquat[body]))


def _friction(model):
    """The book geoms' (sliding, torsional, rolling) friction, one value for all of them."""
    friction = np.unique(model.geom_friction, axis=0)
    if len(friction) != 1:
        raise ValueError(f"the book's geoms have different frictions {friction}")
    return tuple(float(v) for v in friction[0])


@clone
def spawn_book(prim_path, cfg: BookCfg, translation=None, orientation=None, **kwargs):
    """``{prim_path}``: the base rigid body (articulation root) at the book's body frame (centre
    of the back cover's underside); ``{prim_path}/geometry/geom_NN``: its colliders;
    ``{prim_path}/book_cover``: the front cover at its closed pose, with its colliders;
    ``{prim_path}/book_cover_hinge``: the hinge (MuJoCo geom order and names)."""
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    model = scene.book_model()
    root = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)

    sliding, torsional, rolling = _friction(model)
    physics_material = f"{prim_path}/physicsMaterial"
    spawn_physics_material(physics_material, materials.RigidBodyMaterialBaseCfg(
        static_friction=sliding, dynamic_friction=sliding, restitution=0.0), stage=stage)
    material_prim = stage.GetPrimAtPath(physics_material)
    material_prim.CreateAttribute("newton:torsionalFriction", Sdf.ValueTypeNames.Float).Set(torsional)
    material_prim.CreateAttribute("newton:rollingFriction", Sdf.ValueTypeNames.Float).Set(rolling)
    visual_materials = {}

    cover_path = f"{prim_path}/{COVER_PRIM}"
    create_prim(cover_path, prim_type="Xform", stage=stage,
                translation=tuple(float(v) for v in model.body_pos[COVER_BODY]),
                orientation=_xyzw(model.body_quat[COVER_BODY]))
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        if body not in (BASE_BODY, COVER_BODY):
            continue
        if mujoco.mjtGeom(int(model.geom_type[g])) != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"book geom {g} is not a box")
        parent = prim_path if body == BASE_BODY else cover_path
        # A USD Cube of size 2 spans [-1, 1]: scale it by the half-extents.
        prim = create_prim(f"{parent}/geometry/geom_{g:02d}", "Cube", stage=stage,
                           scale=tuple(float(s) for s in model.geom_size[g]), attributes={"size": 2.0},
                           translation=tuple(float(v) for v in model.geom_pos[g]),
                           orientation=_xyzw(model.geom_quat[g]))
        path = str(prim.GetPath())
        if cfg.collision_props is not None:
            schemas.define_collision_properties(path, cfg.collision_props, stage=stage)
        color = tuple(round(float(c), 3) for c in model.geom_rgba[g][:3])
        if color not in visual_materials:
            visual_materials[color] = f"{prim_path}/visualMaterial_{len(visual_materials)}"
            visual = materials.PreviewSurfaceCfg(diffuse_color=color)
            visual.func(visual_materials[color], visual)
        bind_visual_material(path, visual_materials[color], stage=stage)
        bind_physics_material(path, physics_material, stage=stage)

    for path, body in ((prim_path, BASE_BODY), (cover_path, COVER_BODY)):
        _mass(stage.GetPrimAtPath(path), model, body)
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(path, cfg.rigid_props, stage=stage)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    root.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

    # The hinge: its frame is the cover body's frame (MuJoCo joint at the body origin).
    joint_id = model.joint(HINGE).id
    if not np.allclose(model.jnt_axis[joint_id], (0.0, 1.0, 0.0)) or np.any(model.jnt_pos[joint_id]):
        raise ValueError("the book hinge is expected along y at the cover body's origin")
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/{HINGE}")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(prim_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(cover_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in model.body_pos[COVER_BODY])))
    joint.CreateLocalRot0Attr(_quatf(model.body_quat[COVER_BODY]))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateAxisAttr("Y")
    lower, upper = (math.degrees(float(v)) for v in model.jnt_range[joint_id])
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    dof = model.jnt_dofadr[joint_id]
    hinge = joint.GetPrim()
    # Passive: no drive. Newton reads newton:damping per degree (USD's angle unit) and
    # converts it to per radian; friction loss [N m] and armature [kg m^2] carry no angle.
    hinge.CreateAttribute("newton:damping", Sdf.ValueTypeNames.Float).Set(
        float(model.dof_damping[dof]) * math.pi / 180.0)
    hinge.CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(float(model.dof_frictionloss[dof]))
    hinge.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float).Set(float(model.dof_armature[dof]))
    return root


@configclass
class BookCfg(RigidObjectSpawnerCfg):
    """Spawns the Scene's book; masses, inertias, friction and the hinge come from the Scene,
    not from ``mass_props`` or a physics material."""

    func: Callable = spawn_book
    # collision_props and rigid_props: as for the cube (object_imitation_env.py).
