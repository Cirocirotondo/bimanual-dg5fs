#!/usr/bin/env python3
"""Converts both Limb URDFs into the USD assets the training environment spawns.

Like simtoolreal_newton's conversion: fixed base, fixed joints merged (Mount and
Hand base into wrist_3_link, fingertips into link_<f>_4), no self-collision
within a Limb. Drive gains are set at runtime by the environment.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/convert_urdf.py [--force]
"""
import argparse

from bimanual_training import limbs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="convert again even if the USD exists")
    args = parser.parse_args()

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from pxr import Usd, UsdPhysics

    stiffness, damping, _, _ = limbs.drive_gains()
    for side in limbs.SIDES:
        target = limbs.usd_path(side)
        cfg = UrdfConverterCfg(
            asset_path=str(limbs.urdf_path(side)),
            usd_dir=str(target.parent.parent),
            fix_base=True,
            merge_fixed_joints=True,
            self_collision=False,
            force_usd_conversion=bool(args.force),
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=stiffness, damping=damping),
            ),
        )
        converter = UrdfConverter(cfg)
        stage = Usd.Stage.Open(str(converter.usd_path))
        bodies = [p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        joints = [p.GetName() for p in stage.Traverse()
                  if "Joint" in p.GetTypeName() and p.GetTypeName() != "PhysicsFixedJoint"]
        print(f"{side}: {converter.usd_path}\n  {len(bodies)} bodies, {len(joints)} joints")


if __name__ == "__main__":
    main()
