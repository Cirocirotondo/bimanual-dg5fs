"""The basin Object (CONTEXT.md) as one Isaac Lab rigid body.

Isaac Lab spawns single shapes only, so the basin is authored here as a USD
rigid body with one collider prim per geom of the Scene's MuJoCo basin
(bimanual_dg5fs/scene.py: 24 wall boxes, a bottom cylinder, rim and handle
capsules). Geoms, mass and inertia are read from the compiled MuJoCo model, so
the dimensions live in scene.py only; Newton takes the authored mass, centre of
mass and principal inertia as they are (no density integration of overlapping
colliders).
"""
from __future__ import annotations

from collections.abc import Callable

import mujoco
from pxr import Gf, UsdPhysics

from isaaclab.sim import schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass

from bimanual_dg5fs import scene

BASIN_COLOR = (0.35, 0.60, 0.85)    # the Scene's basin rgba


def _xyzw(wxyz):
    return (float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0]))


@clone
def spawn_basin(prim_path, cfg: BasinCfg, translation=None, orientation=None, **kwargs):
    """``{prim_path}``: an Xform rigid body at the basin's body frame (centre of its bottom);
    ``{prim_path}/geometry/geom_NN``: its colliders, in the MuJoCo geom order."""
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    model = scene.basin_model()
    body = 1
    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    geometry = f"{prim_path}/geometry"
    physics_material = f"{geometry}/physicsMaterial"
    spawn_physics_material(physics_material, cfg.physics_material, stage=stage)
    visual_material = f"{geometry}/visualMaterial"
    cfg.visual_material.func(visual_material, cfg.visual_material)

    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body:
            continue
        size, kind = model.geom_size[g], mujoco.mjtGeom(int(model.geom_type[g]))
        pose = {"translation": tuple(float(v) for v in model.geom_pos[g]), "orientation": _xyzw(model.geom_quat[g])}
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            # A USD Cube of size 2 spans [-1, 1]: scale it by the half-extents.
            prim = create_prim(f"{geometry}/geom_{g:02d}", "Cube", scale=tuple(float(s) for s in size),
                               attributes={"size": 2.0}, stage=stage, **pose)
        elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            # MuJoCo: radius and half-length along the geom's z; USD: radius and full length.
            prim = create_prim(f"{geometry}/geom_{g:02d}",
                               "Capsule" if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else "Cylinder",
                               attributes={"radius": float(size[0]), "height": 2.0 * float(size[1]), "axis": "Z"},
                               stage=stage, **pose)
        else:
            raise ValueError(f"basin geom {g} has unsupported MuJoCo type {kind}")
        path = str(prim.GetPath())
        if cfg.collision_props is not None:
            schemas.define_collision_properties(path, cfg.collision_props, stage=stage)
        bind_visual_material(path, visual_material, stage=stage)
        bind_physics_material(path, physics_material, stage=stage)

    # The Scene's mass and inertia, about the MuJoCo body's inertial frame.
    root = stage.GetPrimAtPath(prim_path)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(model.body_mass[body]))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in model.body_ipos[body])))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(v) for v in model.body_inertia[body])))
    w, x, y, z = (float(v) for v in model.body_iquat[body])
    mass.CreatePrincipalAxesAttr(Gf.Quatf(w, x, y, z))
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    return root


@configclass
class BasinCfg(RigidObjectSpawnerCfg):
    """Spawns the Scene's basin; mass and inertia come from the Scene, not from ``mass_props``."""

    func: Callable = spawn_basin
    physics_material: materials.RigidBodyMaterialBaseCfg = materials.RigidBodyMaterialBaseCfg()
    visual_material: materials.VisualMaterialCfg = materials.PreviewSurfaceCfg(diffuse_color=BASIN_COLOR)
    # collision_props and rigid_props: as for the cube (object_imitation_env.py).
