# Training

RL on the bimanual Twin in the style of simtoolreal_newton: Isaac Lab 3 (kit-less)
on Newton / MuJoCo-Warp, PPO from RSL-RL. Sim only. Terms (Reference, Phase,
Training stage, ...) are defined in `../CONTEXT.md`.

All commands run from `bimanual-dg5fs/` with the Isaac Lab interpreter:

    PY="env -u PYTHONPATH deps/IsaacLab/.venv/bin/python"

## Setup (once)

    bash training/setup.sh                       # Isaac Lab at the pinned commit in deps/, packages, USDs
    $PY training/scripts/build_reference.py teleoperation/demonstrations/20261012_093615_lift_cube_trimmed.npz

`convert_urdf.py` turns the Limb URDFs into `training/assets/usd/` (fixed joints
merged); `build_reference.py` turns a Demonstration into `training/references/<name>.npz`
(measured joints, Hand-base poses and fingertips in the world, checked against
the Hand base MuJoCo recorded).

## Stage 1: Motion imitation (no Object)

    $PY training/scripts/evaluate.py --replay                 # feed-forward check that the task is feasible
    $PY training/scripts/train.py --name stage1               # 1024 envs on the desktop, ~0.5 s per iteration
    tensorboard --logdir training/logs
    $PY training/scripts/evaluate.py --checkpoint training/logs/motion_imitation/<run>/model_<N>.pt [--start N]
    $PY training/scripts/evaluate_viser.py        # in the browser: http://localhost:8080

- 50 Hz, one control step per Reference frame; uniform Reference state
  initialization; the episode ends with the Reference.
- Actions (52), per Limb: Hand-base twist in the world frame -> one damped
  least-squares IK step on the arm; hand targets = mean Reference hand pose + 0.15 a.
- Observations (205): per Limb normalised q, targets, dq, Hand-base position and
  6D orientation, fingertips in the Hand-base frame; plus the Phase.
- Rewards: Hand-base keypoints (0.8, sigma 5 cm), hand q / dq, action rates.
  Termination: a Limb's Hand-base keypoint RMS > 8 cm or a hand joint off by
  > 1.35 rad for 5 steps.
- Success: from frame 0 to the end without termination, mean Hand-base error
  < 2 cm and hand joint RMS < 0.1 rad. `--replay` reaches 1.5 cm / 0.004 rad.

PPO uses simtoolreal's hyperparameters ([512, 256] ELU, lr 5e-5 fixed, 24 steps,
5 epochs x 4 minibatches, gamma 0.99, lambda 0.95, entropy 0.001) on RSL-RL's
OnPolicyRunner rather than simtoolreal's own PPO, which is tied to its single-arm env.

`evaluate_viser.py` picks run, checkpoint (or the feed-forward replay) and start
frame in the browser, streams the deterministic rollout, then lets you scrub it:
both Limbs, the Reference as a transparent ghost, Hand-base frames, per-Limb error
curves. "Terminate as in training" off lets a policy run to the end of the Reference.

Physics uses MuJoCo's native contacts (`use_mujoco_contacts=True`): Newton's own
pipeline sizes its buffers from every shape pair the two Limbs could form (3400
contacts, 17340 constraint rows per world, 18 GB at 4096 envs). With MuJoCo's
the run needs 3 GB at 1024 envs and is twice as fast.

## Fleet (tars / case)

`training/fleet/`: `fleet.sh` pushes the project, launches supervised runs in
rootless docker (simtoolreal_newton's `str_newton:v2` image, same Isaac Lab commit),
reports status and pulls results; `supervise_train.sh` restarts a run from its newest
checkpoint until its target; `wait_wave.sh` waits for a wave and pulls it. About
100-125k steps/s per GPU at 4096 envs. How agents run waves: `../GPU_AGENT_BRIEF.md`,
`../FLEET_DISPATCHER.md`, `../WAVE_PLAN.md`; the diary is `../NIGHT_LOG.md`.

Next stages: a fixed Object pose, then a Transform bank.
