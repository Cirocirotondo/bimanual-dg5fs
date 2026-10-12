# One URDF per Limb is the robot model's source of truth

The Reference setup builds its bimanual Scene in MJCF with `<attach>`, which is convenient for MuJoCo teleoperation but has no direct path into our Isaac Lab / Newton training stack (simtoolreal_newton), which consumes merged arm+hand URDFs. We therefore keep one merged URDF per Limb (official UR5e `ur_description` + official Tesollo DG-5F-S with Mount, 20 DOF), composed by hand once and committed, as the single source of truth: the MuJoCo Scene composes the two Limb URDFs at the Base poses, and training will later spawn the same files as two articulations. To change the robot (e.g. the Mount yaw), edit the Limb URDF, never a derived MJCF or USD.

## Considered Options

- **MJCF scene as source of truth (as in the Reference setup)**: rejected, because training would need a second, drifting URDF/USD model.
- **One URDF containing both Limbs**: rejected, because Isaac Lab treats each Limb naturally as its own articulation, and a single tree would need an artificial world root joining the two.
