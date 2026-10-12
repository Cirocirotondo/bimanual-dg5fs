"""Arm IK for both Limbs at once, with mink.

Each tick solves one differential-IK step that moves each Hand base towards its
target, subject to joint position and velocity limits and to collision
avoidance between the two Limbs and between each Hand and the Table. Only the
Arm joints are integrated: the Hand joints are set from Finger retargeting, so
the avoidance constraints see the fingers where they will actually be.
"""
import mink
import mujoco
import numpy as np

from bimanual_dg5fs import scene
from teleoperation.wrist_mapping import Pose

HAND_BASE = "link_base"
# Bodies of a Limb that the other Limb must keep clear of.
NEAR_HAND_BODIES = ("forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link", "link_mount")


def ik_model():
    """The Scene without the Object: the IK only reasons about the Limbs and Table."""
    spec = scene.build_spec()
    spec.delete(spec.body("object"))
    return spec.compile()


def _geoms(model, predicate):
    return [g for g in range(model.ngeom)
            if model.geom_contype[g] or model.geom_conaffinity[g]
            if predicate(model.body(model.geom_bodyid[g]).name)]


def _is_hand(side):
    return lambda name: name.startswith(f"{side}/link_")


def _is_near_hand(side):
    return lambda name: (name.startswith(f"{side}/link_")
                         or name in {f"{side}/{b}" for b in NEAR_HAND_BODIES})


class BimanualIK:
    def __init__(self, dt, max_joint_velocity=1.0, posture_cost=1e-2,
                 min_distance=0.01, detection_distance=0.08, collision_avoidance=True):
        self.model = ik_model()
        self.dt = dt
        self.configuration = mink.Configuration(self.model)
        self.arm_dofs = {s: [self.model.joint(f"{s}/{j}").dofadr[0] for j in scene.ARM_JOINTS]
                         for s in scene.SIDES}
        self.arm_qadr = {s: [self.model.joint(f"{s}/{j}").qposadr[0] for j in scene.ARM_JOINTS]
                         for s in scene.SIDES}
        self.hand_qadr = {s: [self.model.joint(f"{s}/{j}").qposadr[0] for j in scene.HAND_JOINTS]
                          for s in scene.SIDES}

        self.frame_tasks = {
            s: mink.FrameTask(f"{s}/{HAND_BASE}", "body", position_cost=1.0,
                              orientation_cost=0.5, lm_damping=1.0)
            for s in scene.SIDES
        }
        self.posture_task = mink.PostureTask(self.model, cost=posture_cost)

        velocities = {f"{s}/{j}": max_joint_velocity for s in scene.SIDES for j in scene.ARM_JOINTS}
        self.limits = [mink.ConfigurationLimit(self.model), mink.VelocityLimit(self.model, velocities)]
        if collision_avoidance:
            table = [self.model.geom("table").id]
            pairs = [
                (_geoms(self.model, _is_near_hand("left")), _geoms(self.model, _is_near_hand("right"))),
                (_geoms(self.model, _is_hand("left")), table),
                (_geoms(self.model, _is_hand("right")), table),
            ]
            self.limits.append(mink.CollisionAvoidanceLimit(
                self.model, pairs, minimum_distance_from_collisions=min_distance,
                collision_detection_distance=detection_distance))
        self.reset(scene.home_qpos(self.model))

    def reset(self, qpos):
        self.configuration.update(np.asarray(qpos, dtype=float))
        self.posture_task.set_target(self.configuration.q)

    def hand_base_pose(self, side):
        transform = self.configuration.get_transform_frame_to_world(f"{side}/{HAND_BASE}", "body")
        return Pose(transform.translation(), transform.rotation().as_matrix())

    def arm_q(self, side):
        return self.configuration.q[self.arm_qadr[side]].copy()

    def set_arm_q(self, side, q):
        qpos = self.configuration.q.copy()
        qpos[self.arm_qadr[side]] = q
        self.configuration.update(qpos)

    def step(self, targets, hand_q):
        """targets: {side: Pose of the Hand base}; hand_q: {side: (20,)} -> arm q per side."""
        qpos = self.configuration.q.copy()
        for side in scene.SIDES:
            qpos[self.hand_qadr[side]] = hand_q[side]
        self.configuration.update(qpos)

        for side, task in self.frame_tasks.items():
            pose = targets[side]
            task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(pose.rotation), pose.position))
        velocity = mink.solve_ik(self.configuration, [*self.frame_tasks.values(), self.posture_task],
                                 self.dt, "daqp", damping=1e-3, limits=self.limits)

        arm_velocity = np.zeros(self.model.nv)
        for side in scene.SIDES:
            arm_velocity[self.arm_dofs[side]] = velocity[self.arm_dofs[side]]
        self.configuration.integrate_inplace(arm_velocity, self.dt)
        return {side: self.arm_q(side) for side in scene.SIDES}
