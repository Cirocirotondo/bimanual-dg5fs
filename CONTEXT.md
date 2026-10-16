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
A free body in the Scene that the Hands manipulate, chosen per run: a 15 cm cube (default), a basin with two handles, to be lifted by its handles, a hardcover book, to be opened, the basin with a cube beside it, to put the cube in the basin and lift both, or a coffee bag (an industrial 2.5 kg bag of coffee, 35 × 22 × 10 cm, lying flat; in two variants, with or without flat sealed seams at its short ends), to be lifted by its short ends so that it sags in the middle (it is articulated too: rigid segments along its length joined by sprung joints, standing in for a soft body); the book is articulated (its front cover turns on a hinge along the spine), and with the cube the basin is the Object and the cube an extra free body recorded alongside it.
_Avoid_: cube (unqualified), box

**Task environment**:
A training environment built on top of the Scene that adds observations, actions, resets and rewards for one task.
_Avoid_: environment (unqualified), env

### Training

**Reference**:
The per-frame targets derived from one Demonstration (Hand-base poses and Hand joints of both Limbs, and the Object pose, with the joint angles of an articulated Object such as the book's cover) that a policy learns to imitate.
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
A dense set of feasible Object placements inside the training box, each with the Reference retargeted to it (both Arms re-solved by inverse kinematics in the Demonstration's joint family, the Hands unchanged), used to generalise one Demonstration to Generalised imitation.

**Object placement**:
Where the Object starts on the Table, as a planar move of the Demonstration's start pose: a translation along the Table and a yaw about the vertical axis through the Object's centre, the yaw folded into [-45°, +45°] because the cube looks the same every 90°. The whole Reference moves rigidly with it.
_Avoid_: transform (unqualified), randomization, object pose

**Object symmetry**:
One of the 24 rotations that leave the cube unchanged. At every step the Object's orientation is taken as the symmetry-equivalent one closest to the Reference's current orientation, wherever it is used (rewards, observations, the Object frame of the Hand rewards); so is the orientation of the cube beside the basin.
_Avoid_: canonical orientation, flip

**Feasibility map**:
For a regular grid of Object placements, whether the retargeted Reference can be executed: inverse kinematics converges, every joint stays clear of its limits and under its speed limit, and no Limb touches the other Limb or the Table. Each placement records the first criterion it fails.
_Avoid_: reachability map, workspace (unqualified)
