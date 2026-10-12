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

from bimanual_training import limbs

PROTO_ENV = "/World/envs/env_0"
CONTROL_DT = 0.02           # one control step per Reference frame (50 Hz)
SUBSTEPS = 10               # 2 ms physics, the Scene's timestep
OBSERVATIONS_PER_LIMB = 3 * limbs.NUM_JOINTS + 3 + 6 + 15
ACTIONS_PER_LIMB = 6 + limbs.NUM_HAND


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

    reference: str = "lift_cube"
    # Reference state initialization: a uniform start frame; None starts every
    # episode at this frame instead (evaluation).
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

    # --- terminations: any Limb past a threshold for grace_steps steps ----------
    palm_keypoint_threshold_m: float = 0.08
    hand_position_threshold_rad: float = 1.35
    grace_steps: int = 5
