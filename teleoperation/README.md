# Teleoperation (simulation)

Bimanual teleoperation of the Scene: the Operator wears two MANUS Gloves, an
overhead camera tracks the Wrist board on each Glove, and each Limb follows the
Operator's wrist (Arm) and fingers (Hand). Terms are defined in
[../CONTEXT.md](../CONTEXT.md); the architecture is in
[../docs/adr/0002](../docs/adr/0002-teleop-without-ros-pluggable-outputs.md).

It is the simulated counterpart of the UR5 PC pipeline
(`~/simone/SimToolReal/deployment/teleoperation` for the Arm, Tesollo's
`manus_retarget.py` / `dg5f_s_manus_retarget.py` for the Hand), extended to two
Limbs.

```
MANUS client (tcp 8001, finger angles L+R) ─┐
Pose estimator (tcp 5557, Wrist boards)     ├─► core: Finger retargeting + arm IK ─► SimOutput (Scene)
  or MockSource / ReplaySource              ─┘                                      (RealOutput later)
```

## Running

No hardware (scripted wrists and fingers):

```bash
cd bimanual-dg5fs
env -u PYTHONPATH uv run python -m teleoperation.run_sim --source mock
```

With the Gloves and the camera, in three terminals:

```bash
# 1. MANUS client (Nando's ZMQ build): type 1 + Enter for Core Integrated
teleoperation/manus/run_manus_client.sh

# 2. Pose estimator (both Wrist boards, World board frame; one window per camera)
teleoperation/pose_estimation/run.sh

# 3. Teleoperation
env -u PYTHONPATH uv run python -m teleoperation.run_sim            # add --record to save Demonstrations
```

**Object.** The Scene has the 15 cm cube by default. `--object basin` puts a
two-handled basin there instead (round, rim Ø 38 cm, 14 cm high, 0.5 kg, one grip
bar towards each Hand; sizes in `scene.py` `BASIN_*`), for the task of lifting
it by its handles. Turn the wrists palm down and hook the fingers under the bars.
`--object book` puts a closed hardcover book there (17 x 24 x 3.6 cm, 0.68 kg; sizes
in `scene.py` `BOOK_*`), for the task of opening it: the front cover turns on a hinge
along the spine, which lies towards the left Hand, and the covers overhang the pages
by 8 mm at the fore-edge, the lip to lift the cover by. The hinge angle is recorded
as `object_joint_q`.
`--object basin_cube` puts the basin where it is alone and the 15 cm cube on the
Table 5 cm from its rim, towards the Arm bases (`scene.py` `BASIN_CUBE_*`), for the
task of putting the cube in the basin and then lifting the basin. The basin is the
Object (`object_pose`); the cube's pose is recorded as `cube_pose`.
`--object bag` puts a 2.5 kg industrial bag of coffee (35 x 22 x 10 cm) flat where the
basin stands, its long side between the Hands, for the task of lifting it by its short
ends, which are rounded (half elliptic cylinders sticking out 2.5 cm): it sags about 4 cm
in the middle and gives about 5 mm under a squeezing fingertip.
It is six rigid segments joined by sprung bending and twisting joints (a true soft body
could not be trained on), recorded as `object_joint_q`. `--object bag_sealed` adds a
flat, rigid sealed seam (4 x 22 x 1 cm) beyond each short end, for the fingers to close on.

**Scripted Demonstrations.** Without the gloves and camera, `teleoperation.scripted_demo`
plays keyframed Hand-base poses and Hand joints through the same arm IK, Scene and
recorder (`source: scripted` in the `.json`). `--script bag_lift` lifts `bag_sealed`
with both Hands: fingers pointing down beside each seam, fingertips hooked under it and
driven up against the thumb (a pinch), a ~20 cm lift with the Hands closing in as the bag
sags, a hold, then setting it down and back Home:

```bash
env -u PYTHONPATH uv run python -m teleoperation.scripted_demo --script bag_lift --record --video out.mp4
```

`--script bag_lift_plain --object bag` lifts the plain bag from below: palms up, the
fingers bent back like a spatula's edge slide under each end (the plain bag's end segments
are raised underneath, `BAG_END_LIFT`), the thumbs close over the top; the left Hand goes
first and holds its end before the right one slides in.
The Object is saved in the Demonstration's `.json` (`scene.object`), and replays
and `view_demonstration` load the same one.

```bash
env -u PYTHONPATH uv run python -m teleoperation.run_sim --object basin --record
```

The viewer opens with a view from above (your left on the left, the Hands pointing
up) and a horizontal view from in front of the Hands in its bottom-right corner,
to judge heights against the Table and the Object (`--no-inset` to hide it). The
main view can still be moved with the mouse; the corner view is fixed.

Each Limb starts at the Home pose and engages once its Wrist board has been
visible **without interruption for 3 s**: hold that wrist still until
`<side> Limb engaged` appears. That wrist pose becomes the Neutral pose and is
paired with the robot's Home pose. Then type commands in the terminal (Enter after each):

| Command | Effect |
|---|---|
| `c` | engage / clutch both Limbs (engaging: Arm Home, then 3 s of visibility) |
| `lc`, `rc` | engage / clutch the left / right Limb |
| `h` | home: clutch both, Arms back to the Home pose, Hands open |
| `r` | reset the Scene (discards the episode being recorded) |
| `ok`, `fail` | with `--record`: save the episode with that outcome and reset |
| `q` | quit (an unfinished episode is saved with success unknown) |

A clutched Limb holds its Arm and Hand still. When an engaged Limb's Wrist board
is not seen for 0.75 s, the Arm holds still (same Neutral pose, the fingers keep
following the Glove) and continues as soon as the board is back. Only after
2 s without it (`tracking.lost_after_s`) the Limb clutches, its Arm goes back to
the Home pose with the Hand open, and it engages again once the board has been
visible for 3 s: the new wrist pose is paired with the Home pose again. Engaging by command
(`c`, `lc`, `rc`) works the same way. A Limb clutched by command holds still
where it is until you engage it by command.

Replay a Demonstration through the whole pipeline (reproduces its engage/clutch events):

```bash
env -u PYTHONPATH uv run python -m teleoperation.run_sim --source replay --replay teleoperation/demonstrations/demo_<...>.npz
```

## One-time setup

Done on this PC (2026-10-09) unless marked otherwise.

1. **Pose estimator.** It needs numpy < 2, gtsam and a compiled apriltag, so it
   has its own Python 3.10 environment: `teleoperation/pose_estimation/setup.sh`.
   The code is a copy of the UR5 PC's `~/git/robohand/src/tag-pose-estimation`,
   with these changes (all marked `bimanual-dg5fs` in the code):
   - the RealSense wrapper opens only the colour stream (the default config also
     opens the D435i's IMU, which needs extra permissions) and sets the D435i's
     RGB sensor to 3 ms exposure / full gain (with auto exposure moving cubes blur;
     the old code set the exposure on the depth sensor);
   - tags are detected once per frame (not once per board) with 4 threads;
   - new `detection_type: "ippe_tags"` (`tag_pose_estimation/robust_board_pose.py`):
     per-tag IPPE candidates, the pose most tags agree with (false tags, e.g. a
     ChArUco marker decoded as tag16h5 id 1, are dropped), and for an ambiguous
     single tag both poses are published (`alternatives`). The original
     `ransac_with_refinement` jumped by ~40 deg on single small tags.
2. **Cameras**, in priority order in `config/camera.json`:
   - `overhead`: RealSense D435i, serial `242322072500` (the camera of the UR5
     setup), looking down. Main camera.
   - `below`: RealSense D455, serial `138322253845`, looking up at the hands from
     below. Used only for a Wrist board the overhead camera does not see (e.g. palm
     up, the hand covers the cube). Its pose (`calibration/T_world_camera_bottom.npy`)
     comes from the overhead calibration and a recording of a double-sided ChArUco
     board seen by both cameras (see `calibration/T_world_camera_bottom_description.txt`);
     redo it if either camera moves.

   Both RGB sensors run at 3 ms exposure / full gain (no motion blur); the camera
   below also gets local contrast enhancement (CLAHE, `"clahe": true` in
   `config/camera.json`) because it sees the cubes against bright ceiling lights.

   The estimator (`pose_estimation/run_estimator.py`, started by `run.sh`) looks
   for both boards in every overhead frame, then for the missing ones in the latest
   frame of the camera below, and publishes one message per frame with the camera
   each pose came from. For `realsense-viewer`, install
   `ros-humble-librealsense2` (Intel's apt repository is currently mis-signed) and
   `pose_estimation/99-realsense-libusb.rules` into `/etc/udev/rules.d/`. `"type": "D405"` is just the estimator's
   name for its generic RealSense wrapper.
3. **World board -> camera calibration** (redo whenever the camera or the board moves).
   Run `teleoperation/pose_estimation/calibrate_camera.sh`: the window draws the
   *world* axes on the ChArUco board. Turn the board until, seen from where you
   teleoperate, **red (x) points to your left, green (y) towards you, blue (z) up**,
   with the whole board visible, then press `c`. It writes `calibration/T_world_camera.npy`.
   (`--capture` does the same without a window.) This replaces the estimator's
   own `calibrate_extrinsics_single_camera.py`, which uses ArUco functions removed
   in OpenCV 4.11.
4. **Wrist boards.** `config/boards/<side>_wrist_board.json` must match the
   physical tag cube on each Glove. The right one (UR5 PC definition) fits
   (0.4 px reprojection error). The left one, taken from Nando's repo, does NOT
   fit the left cube, and the estimator's interactive builder
   (`build_wrist_board.sh`) produced a wrong one too (tags 1 and 5 swapped, one
   face inside-out: its per-tag poses suffer from the same single-tag ambiguity).
   The current left definition was reconstructed robustly from a 25 s recording
   of the cube turning under the camera (two faces at a time): every pair of
   tags, all four IPPE combinations, the relative pose most frames agree with;
   loop closure 1.1 deg / 3.2 mm, 0.53 px reprojection on 398 multi-tag frames.
   Older versions are kept as `.bak`.
5. **Glove IMU fusion.** With one small tag seen face-on, two poses fit the image;
   the Glove IMU (MANUS skeleton stream, tcp 8000, wrist node) picks the right one
   (`imu_fusion.py`). It needs the rotation between each Glove's IMU and its Wrist
   board (`calibration/imu_mount_<side>.json`, learned automatically from frames
   where the pose is unambiguous, e.g. two faces visible, and saved), plus the IMU's
   yaw reference, re-learned in every session within a few seconds of moving.
   `run_sim` prints `<side>: Glove IMU calibrated against the camera` when active.
6. **MANUS.** `teleoperation/manus/70-manus-hid.rules` must be in `/etc/udev/rules.d/`
   (non-root access to the dongle). The client's MANUS SDK libraries come from
   Git LFS (`git -C /home/simone/projects/robohand/src/manus lfs pull`); the gRPC
   1.28 / protobuf 3.11 libraries it needs are not in Ubuntu 22.04, so copies
   from the UR5 PC are in `teleoperation/manus/lib/` (git-ignored) and
   `run_manus_client.sh` puts them on the library path. Gloves: left
   `5186FAC7`, right `DC8B6BA8`, already calibrated.

## How it works

**Arms** (`wrist_mapping.py`, `arm_ik.py`), per Limb, every 20 ms:

1. The Wrist board pose (World board frame) passes the filter from the UR5
   pipeline: samples jumping more than 8 cm / 35 deg are dropped, the rest are
   low-passed (alpha 0.7 position, 0.85 rotation).
2. The Wrist calibration moves it from the board to the Operator's wrist:
   `p_wrist = p_board + R_board t`. The right Glove uses
   `calibration/right_wrist2board_calibration.json` in metres (the UR5 config
   uses this offset divided by 100 by mistake). The left Glove has no
   calibration yet, so its wrist is taken at the board.
3. Spatial-relative mapping from the Neutral pose, in world axes (the Operator
   stands behind the Arm bases facing -y, so World board axes = Scene axes):
   `p = p_hand0 + s (p_wrist - p_wrist0)`, `R = exp(rotvec(R_wrist R_wrist0^T)) R_hand0`,
   with `s = mapping.position_scale` = **1.5** (10 cm of wrist motion -> 15 cm of robot
   motion; rotations stay 1:1). Override with `run_sim --position-scale 1.0`.
   The Hand base (DG-5F-S `link_base`) is the point that follows the wrist.
4. The target moves at most 0.45 m/s (0.3 m/s of wrist motion at scale 1.5) and 180 deg/s.
5. One mink IK step for both Limbs: joint position limits, 1 rad/s joint
   speed, at least 1 cm between the two Limbs' wrists and Hands, and between
   each Hand and the Table. The Hands may touch the Object.

**Hands** (`finger_retargeting.py`): TESOLLO's per-joint formula on the 20
MANUS ergonomics angles, right-hand signs from the DG-5F-S adapter on the UR5
PC, left-hand signs from TESOLLO's official node, clamped to the DG-5F-S joint
limits. Tests check that the same Glove angles give mirror-image Hands and that
a fist curls towards the palm.

**Simulation** (`outputs.py`): the Arm and Hand targets go to the Scene's
position actuators; physics runs at 500 Hz, control at 50 Hz, in real time.

## Demonstrations

Only with `--record`, in `teleoperation/demonstrations/` (git-ignored): one
`.npz` + `.json` per episode, 50 Hz. Per Side: commanded and measured Arm and
Hand joints, Hand-base target and measured pose, engaged flag, and the raw
inputs (Wrist board pose + confidence, Glove angles); plus the Object pose and
time. The `.json` holds the outcome, the config and the Scene parameters. See
`recorder.py` for the exact keys.

## Files

| | |
|---|---|
| `run_sim.py` | entry point: viewer, countdown, commands, real-time loop |
| `core.py` | per-Limb engage/clutch state, filter -> mapping -> IK, finger targets |
| `inputs.py` | live ZMQ sources (Gloves, Wrist boards), mock and replay sources |
| `wrist_mapping.py` | filter, Wrist calibration, relative mapping, rate limit |
| `arm_ik.py` | bimanual mink IK with collision avoidance |
| `finger_retargeting.py` | MANUS ergonomics -> DG-5F-S joints |
| `outputs.py` | SimOutput (Scene actuators) |
| `recorder.py` | Demonstrations |
| `config/teleop.json` | all teleoperation parameters |
| `config/boards/` | Wrist boards (right: tags 0, 2, 4; left: 1, 3, 5), World board |
| `config/camera.json`, `config/pose_estimation.json` | pose estimator configs |
| `calibration/` | Wrist calibration (right), camera extrinsics (after calibration) |
| `pose_estimation/` | estimator copy, setup / run scripts, `calibrate_camera.py`, `build_wrist_board.sh` |
| `manus/` | MANUS client launcher, udev rule, gRPC/protobuf libraries |

Tests: `tests/teleoperation/` (`env -u PYTHONPATH uv run pytest`).

## Not done yet

- Real-robot output (Hand targets to Tesollo's ROS 2 driver, Arm targets to the
  UR impedance controller).
- Wrist calibration of the left Glove.
- A correct definition of the left Wrist board (step 4 above).
