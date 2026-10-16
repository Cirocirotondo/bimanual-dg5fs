"""Isaac Lab configuration of the Motion imitation Training stage (CONTEXT.md).

Both Limbs, no Table and no Object. Physics mirrors the MuJoCo Scene the
Demonstrations were recorded in (bimanual_dg5fs/scene.py): 2 ms steps,
implicitfast, elliptic cones, gravity compensation on the Limbs, the Scene's
position drives. Task settings follow simtoolreal_newton's motion imitation.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import MujocoRigidBodyPropertiesCfg

from bimanual_dg5fs import scene
from bimanual_training import limbs

PROTO_ENV = "/World/envs/env_0"
CONTROL_DT = 0.02           # one control step per Reference frame (50 Hz)
SUBSTEPS = 10               # 2 ms physics, the Scene's timestep
OBSERVATIONS_PER_LIMB = 3 * limbs.NUM_JOINTS + 3 + 6 + 15
ACTIONS_PER_LIMB = 6 + limbs.NUM_HAND
# Training stage 2: per Limb the Object's centre and 6D orientation in the Hand-base frame.
OBJECT_OBSERVATIONS_PER_LIMB = 3 + 6
# An articulated Object adds, after those, each of its joints' position and velocity and, per
# Limb, the points the fingers lift it by in the Hand-base frame: the book's cover fore-edge
# midpoint; the bag's two short ends (the tips of its rounded ends, or its seams).
GRIP_OBSERVATIONS_PER_LIMB = 3
GRIP_POINTS = {"book": 1, "bag": 2, "bag_sealed": 2}
# A Scene with extra free bodies (scene.EXTRA_OBJECTS: the cube beside the basin) appends,
# last, per extra body and Limb its centre and 6D orientation in the Hand-base frame.
EXTRA_OBJECT_OBSERVATIONS_PER_LIMB = 3 + 6


def object_observation_space(object_name):
    """Training stage 2's observation size with the Object ``object_name`` (bimanual_dg5fs.scene.OBJECTS)."""
    size = 2 * OBSERVATIONS_PER_LIMB + 1 + 2 * OBJECT_OBSERVATIONS_PER_LIMB
    joints = len(scene.OBJECT_JOINTS[object_name])
    if joints:
        size += 2 * joints + 2 * GRIP_OBSERVATIONS_PER_LIMB * GRIP_POINTS[object_name]
    size += 2 * EXTRA_OBJECT_OBSERVATIONS_PER_LIMB * len(scene.EXTRA_OBJECTS.get(object_name, ()))
    return size


def limb_prim(side):
    return f"{side.capitalize()}Limb"


def make_limb_cfg(side) -> ArticulationCfg:
    usd = limbs.usd_path(side)
    if not usd.is_file():
        raise FileNotFoundError(f"{usd} does not exist; run training/scripts/convert_urdf.py once")
    stiffness, damping, effort, armature = limbs.drive_gains()

    def group(names):
        return ImplicitActuatorCfg(
            joint_names_expr=list(names),
            stiffness={n: stiffness[n] for n in names},
            damping={n: damping[n] for n in names},
            effort_limit_sim={n: effort[n] for n in names},
            armature={n: armature[n] for n in names},
        )

    position, rotation = limbs.base_pose(side)
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/" + limb_prim(side),
        spawn=sim_utils.UsdFileCfg(
            spawn_path=f"{PROTO_ENV}/{limb_prim(side)}",
            usd_path=str(usd),
            make_uninstanceable=True,
            # The Scene's Limbs run with gravity compensation (gravcomp = 1).
            rigid_props=MujocoRigidBodyPropertiesCfg(disable_gravity=True, gravcomp=1.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=position, rot=rotation, joint_vel={".*": 0.0}),
        actuators={"arm": group(limbs.ARM_JOINTS), "hand": group(limbs.HAND_JOINTS)},
        joint_ordering=list(limbs.JOINTS),
        soft_joint_pos_limit_factor=1.0,
    )


@configclass
class BimanualMotionImitationEnvCfg(DirectRLEnvCfg):
    decimation: int = 1
    episode_length_s: float = 30.0          # replaced by the Reference length in the environment
    action_space: int = 2 * ACTIONS_PER_LIMB
    observation_space: int = 2 * OBSERVATIONS_PER_LIMB + 1
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        dt=CONTROL_DT,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=400, nconmax=64, iterations=100, ls_iterations=50, solver="newton",
                integrator="implicitfast", cone="elliptic", impratio=10.0,
                # MuJoCo's own collision detection, as in the Scene the Demonstrations were
                # recorded in. It also honours nconmax / njmax per world: Newton's pipeline
                # sizes them from every shape pair the two Limbs could form (3400 contacts,
                # 17340 constraint rows per world, 18 GB at 4096 envs).
                use_mujoco_contacts=True,
            ),
            num_substeps=SUBSTEPS,
            use_cuda_graph=True,
        ),
    )
    # 1024, not simtoolreal's 4096: on the RTX 4090 throughput is already near its
    # ceiling here (512: 19k steps/s, 1024: 24k) and setup peaks at 11.6 GB; 4096
    # passed 17 GB and froze the desktop that shares the GPU.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=3.5, replicate_physics=True)

    # Filled by the environment when None, so the class loads without the USDs.
    left_limb: ArticulationCfg | None = None
    right_limb: ArticulationCfg | None = None

    # Training stage (CONTEXT.md): 1 Motion imitation, 2 Fixed-object imitation.
    stage: int = 1
    reference: str = "lift_cube"
    # Reference state initialization: this fraction of the episodes starts at the
    # Reference's first frame, the rest at a uniform frame. Uniform starts alone leave
    # the opening the least trained part of the Reference: only episodes started
    # before a frame ever see it. (simtoolreal_newton: rsi_early_probability 0.2.)
    rsi_start_probability: float = 0.2
    # Not None: every episode starts at this frame instead (evaluation).
    rsi_start_index: int | None = None

    # --- actions (simtoolreal_newton control) ---------------------------------
    # Hand-base twist in the world frame, at |action| = 1, saturated keeping its direction.
    arm_translation_speed_m_per_s: float = 0.40
    arm_rotation_speed_rad_per_s: float = 1.0
    ik_damping: float = 0.02
    ik_max_joint_delta_rad: float = 0.05
    # Hand targets: default + scale * action (the default is the Reference's mean hand pose).
    hand_action_scale: float = 0.15
    hand_action_clip_rad: float = 100.0
    # Low-pass on the executed action: a_f += alpha * (a - a_f); 1 executes the raw action.
    # At 50 Hz, 0.3 puts the -3 dB point near 2.9 Hz (simtoolreal_newton's lever against
    # a whole-body limit cycle the action-rate rewards alone did not break).
    action_filter_alpha: float = 1.0
    # MuJoCo does not enforce joint velocity limits: applied targets slew at the URDF limit.
    slew_targets_at_velocity_limit: bool = True

    # --- rewards: Gaussians exp(-mse / 2 sigma^2), averaged over the two Limbs --
    palm_keypoint_weight: float = 0.80
    palm_keypoint_std_m: float = 0.05
    hand_position_weight: float = 0.30
    hand_position_std_rad: float = 0.223607
    hand_velocity_weight: float = 0.12
    hand_velocity_std_rad_per_s: float = 1.0
    hand_action_rate_weight: float = 0.12
    hand_action_rate_std: float = 1.0
    arm_action_rate_weight: float = 0.20
    arm_action_rate_std: float = 1.0
    arm_joint_rate_weight: float = 0.05
    arm_joint_rate_std_rad: float = 0.02
    # Penalty on the commanded targets' RMS acceleration (vibration at the drives),
    # -weight * mean over arm/hand of min(acc / scale, clip); off by default. The
    # lift_cube_2 Reference's own RMS is 0.22 (arm) and 0.62 (hand) rad/s^2; a policy
    # chattering at 25 Hz reaches ~30 (arm) and ~300 (hand).
    target_acceleration_weight: float = 0.0
    arm_target_acceleration_scale: float = 10.0
    hand_target_acceleration_scale: float = 100.0
    target_acceleration_clip: float = 3.0

    # --- terminations: any Limb past a threshold for grace_steps steps ----------
    palm_keypoint_threshold_m: float = 0.08
    hand_position_threshold_rad: float = 1.35
    grace_steps: int = 5


@configclass
class BimanualObjectImitationEnvCfg(BimanualMotionImitationEnvCfg):
    """Training stage 2, Fixed-object imitation: the Table and the Object (the
    Scene's cube or basin) are added, the Object starts where the Reference has it,
    and its pose is tracked too (simtoolreal_newton's object rewards and termination).

    The basin (55 colliders) stays within this contact budget: replay and random
    actions peak at 108 contacts and 165 constraint rows per world (2026-10-13).
    So does the book (4 boxes on 2 bodies): 11 contacts at rest, replay peaks at
    57 contacts and 46 rows, random actions at 15 and 33 (2026-10-13).
    basin_cube (the basin and the cube) does not: it raises nconmax to
    basin_cube_nconmax (below). The bags (13 or 15 colliders on 6 bodies, all on
    the Table) raise it to bag_nconmax: 33 contacts and 99 constraint rows per
    world at rest with the Limbs clear of it, random actions the same (2026-10-15);
    no Demonstration has measured a grasp yet.

    Observations: stage 1's 205, then per Limb the Object's centre and 6D
    orientation in the Hand-base frame (18), so a stage-1 policy can be expanded
    with zero weights (scripts/expand_checkpoint.py). An articulated Object (the
    book) appends its hinge angle and velocity (2) and per Limb the cover's
    fore-edge midpoint in the Hand-base frame (6): 231 in all
    (object_observation_space; the environment sets it from the Object). With
    basin_cube (the basin is the Object, the cube an extra free body) the cube's
    centre and 6D orientation in each Hand-base frame (18) come last: 241.
    The bag (bag, bag_sealed) appends its 10 joints' angles and velocities (20)
    and per Limb its two short ends in the Hand-base frame (12): 255.
    """

    stage: int = 2
    observation_space: int = 2 * OBSERVATIONS_PER_LIMB + 1 + 2 * OBJECT_OBSERVATIONS_PER_LIMB

    sim: SimulationCfg = SimulationCfg(
        dt=CONTROL_DT,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                # The Bases rest on the Table (persistent contacts) and the fingers grip
                # the Object: more contact and constraint room than stage 1.
                njmax=1000, nconmax=160, iterations=100, ls_iterations=50, solver="newton",
                integrator="implicitfast", cone="elliptic", impratio=10.0, use_mujoco_contacts=True,
            ),
            num_substeps=SUBSTEPS,
            use_cuda_graph=True,
        ),
    )

    # The Object (bimanual_dg5fs/scene.py OBJECTS): "cube", "basin", "book", "basin_cube", "bag" or
    # "bag_sealed". None takes the one the
    # Reference was demonstrated with (reference.reference_object); the environment writes the
    # resolved name back here, so a run's config.json names it.
    object: str | None = None
    # The Scene's Table and cube (bimanual_dg5fs/scene.py); MuJoCo's default friction. The
    # basin's geometry, mass and inertia come from the Scene (bimanual_training/basin.py), and so
    # do the book's, with its friction and hinge (bimanual_training/book.py; object_friction unused),
    # and the bag's, with its joints and soft contacts (bimanual_training/bag.py).
    object_size_m: float = scene.OBJECT_SIZE
    object_mass_kg: float = scene.OBJECT_MASS
    object_friction: float = 1.0
    table_friction: float = 1.0

    # --- Object tracking (simtoolreal_newton's weights and widths) --------------
    object_position_weight: float = 0.8
    object_position_std_m: float = 0.05
    object_orientation_weight: float = 0.4
    object_orientation_std_rad: float = 0.5
    # Episode ends when the Object is this far from the Reference for grace_steps.
    object_position_threshold_m: float = 0.12

    # --- an articulated Object's joints (the book's hinge, the bag's 10); unused for a rigid one ---
    # Gaussian of the joints' mean squared error against the Reference (their RMS error, squared).
    object_joint_weight: float = 0.6
    object_joint_std_rad: float = 0.25
    # Not None: the episode also ends when a joint is this far off for grace_steps.
    object_joint_threshold_rad: float | None = None
    # Observed: the joint position normalised to its range (-1 closed .. 1), the velocity times this.
    object_joint_velocity_scale: float = 0.2

    # --- the cube beside the basin (Object basin_cube); unused for the other Objects ---
    # Its tracking, as the Object's; its orientation error is taken over the cube's 24
    # symmetries (the one closest to the Reference's current orientation), since it may
    # land in the basin turned by 90 degrees.
    cube_position_weight: float = 0.8
    cube_position_std_m: float = 0.05
    cube_orientation_weight: float = 0.4
    cube_orientation_std_rad: float = 0.5
    # Episode ends when the cube is this far from the Reference for grace_steps.
    cube_position_threshold_m: float = 0.12
    # basin_cube's contact budget per world (replaces the solver's nconmax for this Object
    # only; njmax stays). MuJoCo-Warp counts its broadphase candidates against it too: a
    # 1-env replay peaks at 181 candidates and 131 contacts (cube in the basin, basin
    # lifted), random actions at 161 / 112, constraint rows at 174 (2026-10-14); 160
    # overflowed during the lift.
    basin_cube_nconmax: int = 256
    # The bags' contact budget per world (replaces the solver's nconmax for "bag" and "bag_sealed";
    # njmax stays): measured below.
    bag_nconmax: int = 256


@configclass
class BimanualPlacementImitationEnvCfg(BimanualObjectImitationEnvCfg):
    """Training stage 3, Generalised imitation: stage 2 with the Object placed
    anywhere a Transform bank covers (scripts/build_transform_bank.py), the whole
    Reference moved with it, and the cube's 24 symmetries (Object symmetry) taken
    as the one closest to the Reference's current orientation. Observations and
    rewards keep stage 2's layout and weights."""

    stage: int = 3
    bank: str = "lift_cube_2_box"
    # Mix of placements per episode (simtoolreal_newton's anchored list): this share
    # on the Demonstration's own placement (bank entry 0), the near band (within
    # the distance and yaw below) weighted near_band_weight, the rest uniform.
    demo_share: float = 0.125
    near_band_m: float = 0.05
    near_band_deg: float = 15.0
    near_band_weight: float = 2.0
    # Not None: every episode uses this bank entry (evaluation).
    placement_index: int | None = None
    # The policy also observes the episode's placement (x, y [m], folded yaw [rad]),
    # appended after stage 2's observations: once the Object is in the Hands, nothing
    # else tells which placement the Reference was moved to.
    observe_placement: bool = False
    # The policy works in the placement's frame: it observes the Hand-base poses
    # moved back by the inverse placement and its twist actions are rotated by the
    # placement's yaw; the Arms' joints are observed as their deviation from the
    # moved Reference added to the Demonstration's. A Reference followed exactly
    # then looks (and is commanded) as the Demonstration's own.
    canonical_frame: bool = False
