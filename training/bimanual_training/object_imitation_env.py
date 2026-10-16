"""Fixed-object imitation (Training stage 2, CONTEXT.md): stage 1 plus the Object.

The Scene's Table becomes a static collider and its Object either a rigid object
(the cube, or the basin built from the Scene's geoms, basin.py) or an articulation
(the book, its cover on a passive hinge, book.py; the bag, bag.py); by default the one the
Reference was demonstrated with. Each episode puts the Object where the
Reference has it at the start frame (with the Reference's velocity, so a reset
mid-lift starts in the Demonstration's hands), and the Object's pose joins the
tracking, as in simtoolreal_newton:

* rewards: Gaussians of the Object's position and orientation error against the
  Reference (weights 0.8 / 0.4, widths 5 cm / 0.5 rad), on top of stage 1's;
* termination: the Object more than 12 cm off for grace_steps steps;
* observations: per Limb the Object's centre and 6D orientation in the Hand-base
  frame, appended after stage 1's (so stage-1 checkpoints expand with zeros).

An articulated Object (the book, the bag) also starts with its joints as in the
Reference, and adds, after everything above: the reward Gaussian of its joints' RMS
error against the Reference (weight 0.6, width 0.25 rad), an optional termination
on the worst joint's error, and the observations of its joints (position normalised
to the range, scaled velocity) and per Limb of the points the fingers lift it by in
the Hand-base frame: the book cover's fore-edge midpoint (the lip); the bag's two
short ends (bag.end_points: the tips of its rounded ends, or its seams), the +x end
first. The Object pose terms track its base body.

The bag (bag.py) is 6 rigid segments on 10 sprung passive joints with soft contacts;
the Table gets the Scene's priority 1 so the bag-Table contact stays hard.

The Object does not flip in the Demonstrations, so its orientation error is the
plain rotation angle (simtoolreal's cuboid symmetries are not needed).

basin_cube: the basin is the Object and the cube an extra free body (the Scene's
EXTRA_OBJECTS), spawned as a second rigid object and reset like the Object from
its own Reference track. It adds, after everything above: the Gaussians of the
cube's position and orientation error against its track (cfg cube_*), the
termination on its position error, and per Limb its centre and 6D orientation in
the Hand-base frame. The cube may land in the basin turned by 90 degrees, so its
orientation is the one of its 24 symmetries closest to the Reference's current
orientation (Object symmetry, as in Training stage 3), in rewards and observations.
"""
from __future__ import annotations

import torch
from pxr import Work

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

from bimanual_dg5fs import scene
from bimanual_training.basin import BasinCfg
from bimanual_training.cuboid_symmetry import cuboid_rotation_symmetries
from bimanual_training.motion_imitation_cfg import PROTO_ENV, BimanualObjectImitationEnvCfg, object_observation_space
from bimanual_training.motion_imitation_env import BimanualMotionImitationEnv, gaussian
from bimanual_training.reference import reference_object
from bimanual_training.rotations import (
    normalize_canonical_quaternion,
    quat_conjugate,
    quat_multiply,
    quat_rotate,
    quat_rotate_inverse,
    quat_to_rotation_6d,
)


def rotation_angle(a, b):
    """Angle [rad] of the rotation between xyzw quaternions ``a`` and ``b``."""
    dot = (a * b).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


TRAINED_OBJECTS = ("cube", "basin", "book", "basin_cube", "bag", "bag_sealed")
BASINS = ("basin", "basin_cube")
BAGS = ("bag", "bag_sealed")


def fore_edge_in_cover():
    """``(3,)`` the book cover's fore-edge midpoint in the cover body frame (scene.book_model)."""
    import numpy as np

    model = scene.book_model()
    geom = model.geom("book_front_cover").id
    if not np.allclose(model.geom_quat[geom], (1.0, 0.0, 0.0, 0.0)):
        raise ValueError("the book cover geom is expected aligned with its body")
    # The cover runs from the spine (the body origin) towards -x, to the fore-edge.
    position = model.geom_pos[geom]
    return torch.tensor([position[0] - model.geom_size[geom][0], position[1], position[2]])


class BimanualObjectImitationEnv(BimanualMotionImitationEnv):
    cfg: BimanualObjectImitationEnvCfg

    def __init__(self, cfg: BimanualObjectImitationEnvCfg, render_mode: str | None = None, **kwargs):
        demonstrated = reference_object(cfg.reference)
        if cfg.object is None:
            cfg.object = demonstrated
        if cfg.object not in TRAINED_OBJECTS:
            raise ValueError(f"Object {cfg.object!r} is not supported for training; choose one of {TRAINED_OBJECTS}")
        if cfg.object != demonstrated:
            print(f"[object imitation] warning: Object {cfg.object!r}, but the Reference {cfg.reference!r} "
                  f"was demonstrated with the {demonstrated}", flush=True)
        self.object_joint_names = scene.OBJECT_JOINTS[cfg.object]
        # The same joints as the articulation names them (Newton merges the bag's two hinges at a
        # junction into one D6 joint, bag.backend_joint_names).
        if cfg.object in BAGS:
            from bimanual_training.bag import backend_joint_names

            self.object_backend_joint_names = tuple(backend_joint_names(cfg.object))
        else:
            self.object_backend_joint_names = self.object_joint_names
        self.articulated = bool(self.object_joint_names)
        # The Scene's other free bodies (the cube beside the basin), name -> RigidObject.
        self.extra_object_names = tuple(scene.EXTRA_OBJECTS.get(cfg.object, ()))
        if set(self.extra_object_names) - {"cube"}:
            raise ValueError(f"extra free bodies {self.extra_object_names}: only the cube is supported")
        self.extra_objects = {}
        if self.articulated or self.extra_object_names:
            cfg.observation_space = object_observation_space(cfg.object)
        # Before the simulation (and its MuJoCo-Warp solver) is created from cfg.sim.
        if self.extra_object_names:
            cfg.sim.physics.solver_cfg.nconmax = cfg.basin_cube_nconmax
        if cfg.object in BAGS:
            cfg.sim.physics.solver_cfg.nconmax = cfg.bag_nconmax
        self.cube_symmetries = cuboid_rotation_symmetries((cfg.object_size_m / 2.0,) * 3).to(
            str(cfg.sim.device), torch.float32)
        super().__init__(cfg, render_mode, **kwargs)
        if self.articulated:
            self._resolve_object_joints()

    # ------------------------------------------------------------------ scene

    def _object_spawn_cfg(self):
        cfg = self.cfg
        material = RigidBodyMaterialBaseCfg(static_friction=cfg.object_friction,
                                            dynamic_friction=cfg.object_friction, restitution=0.0)
        if cfg.object in BASINS:
            return BasinCfg(spawn_path=f"{PROTO_ENV}/Object", physics_material=material,
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                            collision_props=sim_utils.CollisionPropertiesCfg())
        return self._cube_spawn_cfg("Object")

    def _cube_spawn_cfg(self, prim):
        """The Scene's cube (OBJECT_SIZE, OBJECT_MASS) at ``{env_0}/{prim}``: the Object, or the
        extra free body beside the basin."""
        cfg = self.cfg
        material = RigidBodyMaterialBaseCfg(static_friction=cfg.object_friction,
                                            dynamic_friction=cfg.object_friction, restitution=0.0)
        return sim_utils.CuboidCfg(
            spawn_path=f"{PROTO_ENV}/{prim}",
            size=(cfg.object_size_m,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=cfg.object_mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.30, 0.25)),
        )

    def _spawn_extras(self):
        cfg = self.cfg
        half = [s / 2.0 for s in scene.TABLE_SIZE]
        table = sim_utils.CuboidCfg(
            size=tuple(scene.TABLE_SIZE),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=cfg.table_friction,
                                                      dynamic_friction=cfg.table_friction, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35)),
        )
        table.func(f"{PROTO_ENV}/Table", table,
                   translation=(scene.TABLE_CENTER_XY[0], scene.TABLE_CENTER_XY[1], scene.TABLE_TOP_Z - half[2]))
        if cfg.object in BAGS:
            # As in the Scene: the Table keeps its own hard contact with the soft bag (priority 1).
            from pxr import Usd, UsdPhysics

            from isaaclab.sim.utils import get_current_stage

            from bimanual_training.bag import harden_against_bag

            stage = get_current_stage()
            for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{PROTO_ENV}/Table")):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    harden_against_bag(stage, str(prim.GetPath()))
        if cfg.object in BASINS and Work.GetConcurrencyLimit() != 1:
            # launch.py explains; set by importing bimanual_training.launch before pxr.
            raise RuntimeError("the basin needs PXR_WORK_THREAD_LIMIT=1 set before pxr is imported "
                               "(bimanual_training/launch.py): parallel USD parsing of its 55 colliders "
                               "corrupts the heap")
        start = self.reference.object_pose[0].tolist()
        if self.articulated:
            self._spawn_articulated_object(start)
            return
        self.object = RigidObject(RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=self._object_spawn_cfg(),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(start[:3]), rot=tuple(start[3:7])),
        ))
        self._spawn_extra_objects()

    def _spawn_extra_objects(self):
        missing = [name for name in self.extra_object_names if name not in self.reference.extra_objects]
        if missing:
            raise ValueError(f"the Reference {self.cfg.reference!r} has no track of {missing}: rebuild it with "
                             "scripts/build_reference.py from a Demonstration of the "
                             f"{self.cfg.object}")
        for name in self.extra_object_names:
            start = getattr(self.reference, f"{name}_pose")[0].tolist()
            prim = name.capitalize()
            self.extra_objects[name] = RigidObject(RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + prim,
                spawn=self._cube_spawn_cfg(prim),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(start[:3]), rot=tuple(start[3:7])),
            ))

    def _spawn_articulated_object(self, start):
        """The book or the bag: its base free-floating, its joints passive (no actuator, no drive)."""
        from bimanual_training.bag import BagCfg
        from bimanual_training.book import BookCfg

        if self.reference.object_joints != len(self.object_joint_names):
            raise ValueError(f"the Reference {self.cfg.reference!r} has {self.reference.object_joints} Object joints, "
                             f"the {self.cfg.object} {list(self.object_joint_names)}: rebuild it with "
                             "scripts/build_reference.py")
        joint_q = self.reference.object_joint_q[0].tolist()
        self.object = Articulation(ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=(BagCfg(spawn_path=f"{PROTO_ENV}/Object", object_name=self.cfg.object,
                          rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                          collision_props=sim_utils.CollisionPropertiesCfg())
                   if self.cfg.object in BAGS else
                   BookCfg(spawn_path=f"{PROTO_ENV}/Object", rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                           collision_props=sim_utils.CollisionPropertiesCfg())),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(start[:3]), rot=tuple(start[3:7]),
                joint_pos={f"^{name}$": q for name, q in zip(self.object_backend_joint_names, joint_q)},
                joint_vel={".*": 0.0}),
            actuators={},
            joint_ordering=list(self.object_backend_joint_names),
            soft_joint_pos_limit_factor=1.0,
        ))

    def _resolve_object_joints(self):
        if list(self.object.joint_names) != list(self.object_backend_joint_names):
            raise ValueError(f"Object joints {self.object.joint_names}, expected {self.object_backend_joint_names}")
        limits = self.object.data.joint_pos_limits.torch[0].clone()
        self.object_joint_lower, self.object_joint_upper = limits[:, 0], limits[:, 1]
        # Each joint is the Scene's (its range identifies it: the bag's bends and twists differ).
        scene_ranges = torch.tensor([scene.object_joint_ranges(self.cfg.object)[name] for name in self.object_joint_names],
                                    device=limits.device, dtype=limits.dtype)
        if not torch.allclose(limits, scene_ranges, atol=1e-4):
            raise ValueError(f"Object joint ranges {limits.tolist()}, the Scene's {scene_ranges.tolist()}")
        # The points the fingers lift the Object by, each on one of its bodies (GRIP_POINTS in
        # motion_imitation_cfg): the book's cover fore-edge; the bag's two short ends.
        if self.cfg.object in BAGS:
            from bimanual_training.bag import end_points

            grips = end_points(self.cfg.object)
        else:
            grips = [("book_cover", fore_edge_in_cover())]
        self.grip_bodies = [self.object.find_bodies(f"^{body}$")[0][0] for body, _ in grips]
        self.grip_in_body = torch.stack([point for _, point in grips]).to(self.device, torch.float32)   # (K, 3)

    def _register_extras(self):
        if self.articulated:
            self.scene.articulations["object"] = self.object
        else:
            self.scene.rigid_objects["object"] = self.object
        for name, body in self.extra_objects.items():
            self.scene.rigid_objects[name] = body

    def _reset_extras(self, env_ids, start):
        pose = self.ref("object_pose", start, env_ids).clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        ids = env_ids.to(torch.int32)
        self.object.write_root_pose_to_sim_index(root_pose=pose.contiguous(), env_ids=ids)
        self.object.write_root_velocity_to_sim_index(
            root_velocity=self.ref("object_velocity", start, env_ids).contiguous(), env_ids=ids)
        if self.articulated:
            self.object.write_joint_state_to_sim_index(
                position=self.ref("object_joint_q", start, env_ids).contiguous(),
                velocity=self.ref("object_joint_dq", start, env_ids).contiguous(), env_ids=ids)
        for name, body in self.extra_objects.items():
            pose = self.ref(f"{name}_pose", start, env_ids).clone()
            pose[:, :3] += self.scene.env_origins[env_ids]
            body.write_root_pose_to_sim_index(root_pose=pose.contiguous(), env_ids=ids)
            body.write_root_velocity_to_sim_index(
                root_velocity=self.ref(f"{name}_velocity", start, env_ids).contiguous(), env_ids=ids)

    # ------------------------------------------------------------------ state

    def object_pose(self):
        """Object position ``(N, 3)`` in the env frame and orientation ``(N, 4)`` xyzw."""
        pose = self.object.data.root_link_pose_w.torch
        return pose[:, 0:3] - self.scene.env_origins, normalize_canonical_quaternion(pose[:, 3:7])

    def cube_pose(self):
        """The cube beside the basin: position ``(N, 3)`` in the env frame and orientation ``(N, 4)``
        xyzw, the one of its 24 symmetries closest to the Reference's current orientation."""
        pose = self.extra_objects["cube"].data.root_link_pose_w.torch
        quat = normalize_canonical_quaternion(pose[:, 3:7])
        reference = normalize_canonical_quaternion(self.ref("cube_pose", self.reference_index)[:, 3:7])
        count = len(self.cube_symmetries)
        candidates = quat_multiply(quat.unsqueeze(1).expand(-1, count, -1),
                                   self.cube_symmetries.unsqueeze(0).expand(len(quat), -1, -1))     # (N, 24, 4)
        best = (candidates * reference.unsqueeze(1)).sum(-1).abs().argmax(dim=1)
        chosen = candidates[torch.arange(len(quat), device=self.device), best]
        return pose[:, 0:3] - self.scene.env_origins, normalize_canonical_quaternion(chosen)

    def object_joint_q(self):
        """``(N, J)`` an articulated Object's joint positions [rad]."""
        return self.object.data.joint_pos.torch

    def grip_points(self):
        """``(N, K, 3)`` the points the fingers lift an articulated Object by, in the env frame: the book
        cover's fore-edge midpoint (K = 1); the bag's +x and -x short ends (K = 2)."""
        bodies = self.object.data.body_link_pose_w.torch[:, self.grip_bodies]                        # (N, K, 7)
        quat = normalize_canonical_quaternion(bodies[..., 3:7])
        return (bodies[..., 0:3] - self.scene.env_origins.unsqueeze(1)
                + quat_rotate(quat, self.grip_in_body.expand_as(bodies[..., 0:3])))

    @staticmethod
    def _in_hand_frames(position, quat, palm_position, palm_quat):
        """``(N, 2 * 9)``: a body's centre and 6D orientation in each Limb's Hand-base frame."""
        centre = quat_rotate_inverse(palm_quat, position.unsqueeze(1) - palm_position)                 # (N, 2, 3)
        orientation = quat_multiply(quat_conjugate(palm_quat), quat.unsqueeze(1).expand_as(palm_quat))
        per_limb = torch.cat((centre, quat_to_rotation_6d(normalize_canonical_quaternion(orientation))), dim=-1)
        return per_limb.flatten(1)

    def _extra_observations(self):
        object_position, object_quat = self.object_pose()
        palm_position, palm_quat = self.palm_pose()
        observation = self._in_hand_frames(object_position, object_quat, palm_position, palm_quat)
        if self.extra_objects:
            # Last, after every other Object's layout: a stage-1 checkpoint still expands with zeros.
            observation = torch.cat((observation, self._in_hand_frames(*self.cube_pose(), palm_position, palm_quat)),
                                    dim=1)
        if not self.articulated:
            return observation
        # Appended after the rigid Objects' layout, so a stage-1 checkpoint still expands with zeros.
        q = self.object_joint_q()
        normalised = (2.0 * (q - self.object_joint_lower) / (self.object_joint_upper - self.object_joint_lower)
                      - 1.0).clamp(-1.0, 1.0)
        velocity = self.object.data.joint_vel.torch * self.cfg.object_joint_velocity_scale
        grips = self.grip_points()                                                                    # (N, K, 3)
        count = grips.shape[1]
        lip = quat_rotate_inverse(palm_quat.unsqueeze(2).expand(-1, -1, count, -1),
                                  grips.unsqueeze(1) - palm_position.unsqueeze(2))                    # (N, 2, K, 3)
        return torch.cat((observation, normalised, velocity, lip.flatten(1)), dim=1)

    def _object_errors(self):
        index = self.reference_index
        position, quat = self.object_pose()
        reference = self.ref("object_pose", index)
        position_error = (position - reference[:, :3]).norm(dim=-1)
        orientation_error = rotation_angle(quat, normalize_canonical_quaternion(reference[:, 3:7]))
        return position_error, orientation_error

    def _cube_errors(self):
        position, quat = self.cube_pose()
        reference = self.ref("cube_pose", self.reference_index)
        return (position - reference[:, :3]).norm(dim=-1), rotation_angle(quat, normalize_canonical_quaternion(
            reference[:, 3:7])), position

    # --------------------------------------------------- rewards, terminations

    def _compute_metrics(self):
        reward, metrics, palm_error, hand_error = super()._compute_metrics()
        cfg = self.cfg
        position_error, orientation_error = self._object_errors()
        position_reward = gaussian(position_error.square(), cfg.object_position_std_m)
        orientation_reward = gaussian(orientation_error.square(), cfg.object_orientation_std_rad)
        reward = reward + cfg.object_position_weight * position_reward + cfg.object_orientation_weight * orientation_reward
        object_height = self.object_pose()[0][:, 2]
        metrics.update({
            "reward_object_position": position_reward,
            "reward_object_orientation": orientation_reward,
            "object_position_error_m": position_error,
            "object_orientation_error_rad": orientation_error,
            "object_height_m": object_height,
        })
        self._object_position_error = position_error
        if self.articulated:
            joint_error = self.object_joint_q() - self.ref("object_joint_q", self.reference_index)       # (N, J)
            joint_reward = gaussian(joint_error.square().mean(-1), cfg.object_joint_std_rad)
            reward = reward + cfg.object_joint_weight * joint_reward
            self._object_joint_error = joint_error.abs().amax(dim=-1)
            metrics.update({
                "reward_object_joint": joint_reward,
                "object_joint_error_rad": self._object_joint_error,
                "object_joint_q_rad": self.object_joint_q()[:, 0],
            })
        if self.extra_objects:
            cube_error, cube_angle, cube_position = self._cube_errors()
            cube_position_reward = gaussian(cube_error.square(), cfg.cube_position_std_m)
            cube_orientation_reward = gaussian(cube_angle.square(), cfg.cube_orientation_std_rad)
            reward = (reward + cfg.cube_position_weight * cube_position_reward
                      + cfg.cube_orientation_weight * cube_orientation_reward)
            basin_position = self.object_pose()[0]
            metrics.update({
                "reward_cube_position": cube_position_reward,
                "reward_cube_orientation": cube_orientation_reward,
                "cube_position_error_m": cube_error,
                "cube_orientation_error_rad": cube_angle,
                "cube_height_m": cube_position[:, 2],
                # The cube's centre above the basin's bottom and its horizontal distance from the
                # basin's axis: in the basin when under the bottom radius and above the bottom.
                "cube_above_basin_m": cube_position[:, 2] - basin_position[:, 2],
                "cube_basin_distance_m": (cube_position[:, :2] - basin_position[:, :2]).norm(dim=-1),
            })
            self._cube_position_error = cube_error
        return reward, metrics, palm_error, hand_error

    def _extra_violation(self):
        violation = self._object_position_error > self.cfg.object_position_threshold_m
        if self.articulated and self.cfg.object_joint_threshold_rad is not None:
            violation = violation | (self._object_joint_error > self.cfg.object_joint_threshold_rad)
        # A blown-up Object ends its episode like a blown-up Limb does.
        pose = self.object.data.root_link_pose_w.torch
        finite = torch.isfinite(pose).all(dim=1)
        if self.articulated:
            finite = finite & torch.isfinite(self.object_joint_q()).all(dim=1)
        if self.extra_objects:
            violation = violation | (self._cube_position_error > self.cfg.cube_position_threshold_m)
            for body in self.extra_objects.values():
                finite = finite & torch.isfinite(body.data.root_link_pose_w.torch).all(dim=1)
        return violation | ~finite
