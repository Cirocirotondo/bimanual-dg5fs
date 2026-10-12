# Bimanual DG-5F-S

A twin of Nando's bimanual teleoperation setup in which each UR5e arm carries a Tesollo DG-5F-S hand, built for teleoperation and policy training.

## Language

### Setups

**Reference setup**:
Nando's original bimanual rig and its repo (`robohand-master_wsl3`): two UR5e arms with placeholder end-effectors, driven by cube tracking and Manus gloves.
_Avoid_: original repo, Nando's repo

**Twin**:
This project: a clean sibling of the Reference setup that keeps its layout, arm placement and conventions, but mounts a Hand on each Arm.
_Avoid_: copy, fork, clone

### Robot

**Side**:
Which half of the bimanual rig something belongs to, left or right, as seen by the operator of this Twin (opposite to the Reference setup's viewpoint, so the Reference "left" Arm is our right one); an Arm and the Hand mounted on it always share a Side.
_Avoid_: ur5_1 / ur5_2 and the Reference's left/right labels

**Arm**:
One UR5e manipulator, identified by its Side.
_Avoid_: robot, UR

**Hand**:
One 20-DOF Tesollo DG-5F-S dexterous hand, left- or right-handed to match its Side.
_Avoid_: gripper, end-effector, DG5F (that is the larger model), 15-DOF variant

**Mount**:
Tesollo's adapter that fixes a Hand to its Arm's tool flange.
_Avoid_: adapter plate, flange plate

**Mount yaw**:
The rotation of a Hand about its Arm's tool axis; set per Side and still to be measured on the real rig.

**Limb**:
One Arm with its Hand fixed on it by the Mount, for one Side; the unit that is modelled, simulated and commanded as a whole.
_Avoid_: robot, arm unit, arm (when the Hand is included)

**Base pose**:
The world pose of an Arm's base: the Reference setup's calibrated poses, made mirror-symmetric between the two Sides.

**Home pose**:
The joint configuration both Limbs start from: for the right Arm, the Reference setup's default teleop home with the wrist turned so the Hand points forward, thumb up, instead of at the other Hand; the left Limb is its mirror image; both Hands fully open.
_Avoid_: rest pose, default pose, initial pose

### Teleoperation

**Operator**:
The person teleoperating, standing behind the Arm bases and facing the way the Hands point; Sides are named from the Operator's point of view.

**Glove**:
One MANUS glove worn by the Operator; it supplies the finger angles for the Hand of the same Side.
_Avoid_: manus (unqualified)

**Wrist board**:
The small AprilTag cube strapped to a Glove's wrist (right: tags 0, 2, 4; left: tags 1, 3, 5), whose tracked pose gives the Operator's wrist pose.
_Avoid_: tracking cube (the Reference setup's hand-held input device), hand board

**World board**:
The ChArUco board that defines the tracking world frame in which the overhead camera reports Wrist board poses.
_Avoid_: charuco (unqualified), calibration board

**Wrist calibration**:
The fixed transform from a Wrist board to the Operator's wrist on that Glove.
_Avoid_: board offset, tracking origin offset

**Neutral pose**:
The Operator's wrist pose captured when a Limb is engaged, paired with the robot's Home pose; afterwards the Limb moves away from Home by how far the wrist has moved from it.
_Avoid_: initial pose, zero pose, home (that is the robot's Home pose)

**Clutch**:
Disengaging a Limb so it holds still; re-engaging sends it back to the Home pose and captures a new Neutral pose.
_Avoid_: pause (unqualified), freeze

**Finger retargeting**:
Turning a Glove's finger angles into Hand joint targets with Tesollo's per-joint formula.
_Avoid_: hand IK, finger mapping

**Demonstration**:
One recorded teleoperation episode in the Scene, saved only when recording is requested.
_Avoid_: recording, trajectory, log

### Simulation

**Scene**:
The simulated world containing both Limbs at their Base poses, the Table and the Object; it can be loaded, rendered and moved, but defines no task.
_Avoid_: environment, env, world

**Table**:
The work surface the Limbs act on in the Scene.

**Object**:
A free rigid body in the Scene that the Hands manipulate; currently a single 15 cm cube.
_Avoid_: cube (unqualified), box

**Task environment**:
A training environment built on top of the Scene that adds observations, actions, resets and rewards for one task.
_Avoid_: environment (unqualified), env

### Training

**Reference**:
The per-frame targets derived from one Demonstration (Hand-base poses and Hand joints of both Limbs, and the Object pose) that a policy learns to imitate.
_Avoid_: target trajectory, expert data, demo (unqualified)

**Phase**:
How far into the Reference an episode is, from 0 at its first frame to 1 at its last; the only part of the Reference the policy observes.
_Avoid_: progress, time index

**Reference state initialization**:
Starting a training episode at a random frame of the Reference, with the Limbs (and the Object) placed as in that frame.
_Avoid_: random start, RSI (unexpanded)

**Training stage**:
One of three successive versions of the imitation task: Motion imitation (the Limbs reproduce the Reference without the Object), Fixed-object imitation (the Object starts exactly as in the Demonstration) and Generalised imitation (the Object starts anywhere a Transform bank covers).
_Avoid_: curriculum level, phase (that is the Phase)

**Transform bank**:
A set of Object start poses, each with the Reference retargeted to it, used to generalise one Demonstration to other Object placements.
