#!/usr/bin/env python3
"""A scripted Demonstration: the Hand bases follow keyframed poses instead of the
Operator's wrists, through the same arm IK, Scene and recorder as teleoperation.

The script is a list of keyframes (time [s], per side Hand-base position and
orientation, Hand joints). Between keyframes the targets follow a minimum-jerk
profile (orientations by slerp). The episode is recorded like a teleoperated one (engaged
throughout; no Glove or Wrist inputs), so it goes through trim_demonstration,
build_reference and training unchanged.

    env -u PYTHONPATH uv run python -m teleoperation.scripted_demo --script bag_lift [--object bag] \
        [--record] [--video /path/to/out.mp4] [--param lift=0.25 ...]
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from bimanual_dg5fs import scene
from teleoperation.arm_ik import BimanualIK
from teleoperation.core import LimbTargets
from teleoperation.inputs import Inputs
from teleoperation.outputs import SimOutput
from teleoperation.recorder import DemonstrationRecorder
from teleoperation.wrist_mapping import Pose

CONTROL_HZ = 50.0
DEMONSTRATIONS = Path(__file__).resolve().parent / "demonstrations"


@dataclass
class Keyframe:
    time: float                     # [s]
    position: dict                  # {side: (3,)} Hand-base position [m]
    hand_q: dict                    # {side: (20,)} Hand joints [rad]
    rotation: dict = None           # {side: (3, 3)} Hand-base orientation; None: Home


def minimum_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)


def side_sign(side):
    return 1.0 if side == "left" else -1.0              # the left Limb is on +x


def fingers_down(side):
    """Hand-base orientation with the fingers pointing down and the palm facing the bag's centre
    (the Hand base's x axis is the palm normal, z the finger direction)."""
    palm = np.array([-side_sign(side), 0.0, 0.0])
    fingers = np.array([0.0, 0.0, -1.0])
    return np.column_stack([palm, np.cross(fingers, palm), fingers])


def hook(p, side, thumb_press=0.0, finger_press=0.0):
    """Hand joints of the pinch: fingers 2-5 hooked under the seam (bent at their middle and distal
    joints), the thumb brought just over it. ``finger_press`` bends the fingers' middle joints further,
    driving the fingertips up into the seam so that it is squeezed against the thumb; ``thumb_press``
    drives the thumb down past the seam's top.
    The thumb's joints are mirrored between the Hands (the right one's run the other way), the
    fingers' are not."""
    bend = {"2": p["hook_proximal"], "3": p["hook_middle"] + finger_press, "4": p["hook_distal"]}
    mirror = side_sign(side)
    thumb = {str(k): p[f"thumb_{k}"] for k in (1, 2, 3, 4)}
    thumb["3"] -= thumb_press
    thumb = {k: mirror * v for k, v in thumb.items()}
    q = []
    for name in scene.HAND_JOINTS:
        finger, joint = name.split("_")[1], name.split("_")[2]
        q.append(thumb[joint] if finger == "1" else bend.get(joint, 0.0))
    return np.array(q)


def bag_lift(home, home_rotation, p):
    """Lift bag_sealed by its seams: each Hand comes down beside a seam with its fingers pointing
    down, hooks the fingertips under it, pinches it against the thumb from above, lifts, holds, sets
    the bag down, lets go and goes Home.

    ``home``/``home_rotation``: {side: Home Hand-base position / orientation}; ``p``: BAG_LIFT_PARAMETERS."""
    open_hand = {s: np.zeros(len(scene.HAND_JOINTS)) for s in scene.SIDES}
    hooked = {s: hook(p, s) for s in scene.SIDES}
    pinched = {s: hook(p, s, p["thumb_press"], p["finger_press"]) for s in scene.SIDES}
    down = {s: fingers_down(s) for s in scene.SIDES}

    def at(x, z, y=p["y"]):
        return {s: np.array([scene.BAG_XY[0] + side_sign(s) * x, y, z]) for s in scene.SIDES}

    low, high = p["z"], p["z"] + p["lift"]
    held = p["x"] - p["follow"]          # the Hands close in while lifting: the sagging bag gets shorter
    return [
        Keyframe(0.0, dict(home), open_hand, dict(home_rotation)),
        Keyframe(1.0, dict(home), open_hand, dict(home_rotation)),
        Keyframe(3.5, at(p["x_clear"], low + 0.10), open_hand, down),               # above, outside the seams
        Keyframe(5.0, at(p["x_clear"], low), open_hand, down),                      # fingers beside the seams
        Keyframe(6.0, at(p["x"], low), hooked, down),                               # hook under the seams
        Keyframe(7.0, at(p["x"], low), pinched, down),                              # pinch with the thumbs
        Keyframe(11.0, at(held, high), pinched, down),                              # lift
        Keyframe(13.0, at(held, high), pinched, down),                              # hold
        Keyframe(16.5, at(p["x"], low + p["set_down"]), pinched, down),             # set it down
        Keyframe(17.0, at(p["x"], low + p["set_down"]), hooked, down),              # thumbs off
        Keyframe(18.0, at(p["x_clear"], low + p["set_down"]), open_hand, down),     # unhook
        Keyframe(19.5, at(p["x_clear"], low + 0.10), open_hand, down),
        Keyframe(22.0, dict(home), open_hand, dict(home_rotation)),
        Keyframe(23.0, dict(home), open_hand, dict(home_rotation)),
    ]


BAG_LIFT_PARAMETERS = {
    "y": -0.55,            # Hand-base y: the fingers across the middle of the seam's width
    "z": 0.205,            # Hand-base height while hooked (fingertips ~1 cm under the seam)
    "x": 0.263,            # Hand-base distance from the bag's centre while hooked (fingertips near the seam's root)
    "x_clear": 0.29,       # ... while coming down beside the seams (fingertips outside them)
    "follow": 0.03,        # how far the Hands close in while lifting [m]
    "hook_proximal": 0.0,  # finger joints _2, _3, _4 of the hook [rad]
    "hook_middle": 0.9,
    "hook_distal": 1.4,
    "finger_press": 0.7,   # extra middle-joint bend squeezing the seam up against the thumb [rad]
    "thumb_1": -1.25,      # left thumb joints _1.._4: its tip just over the seam [rad] (right: mirrored)
    "thumb_2": 0.3,
    "thumb_3": -0.5,
    "thumb_4": 0.0,
    "thumb_press": 0.0,    # the thumb driven past the seam's top [rad] (it cannot reach much lower)
    "lift": 0.25,          # Hand-base lift [m]: the arms reach ~0.20 near full stretch, the bag's centre rises ~0.17
    "set_down": 0.005,     # set the bag down this much above where it was lifted from [m]
}
def spatula(side, pitch):
    """Hand-base orientation of the scoop: palm up, the fingers pointing at the bag's centre,
    tilted down by ``pitch`` [rad] so that the palm stays clear of the Table."""
    fingers = np.array([-side_sign(side) * np.cos(pitch), 0.0, -np.sin(pitch)])
    palm = np.array([-side_sign(side) * np.sin(pitch), 0.0, np.cos(pitch)])
    return np.column_stack([palm, np.cross(fingers, palm), fingers])


def scoop_hand(p, side, thumb=True):
    """Hand joints of the scoop: fingers 2-5 bent back at their middle and distal joints (the tips down
    to the Table, a spatula's edge), the thumb over the bag's top face (``thumb``) or straight."""
    mirror = side_sign(side)
    q = []
    for name in scene.HAND_JOINTS:
        finger, joint = name.split("_")[1], name.split("_")[2]
        if finger == "1":
            q.append(mirror * p[f"thumb_{joint}"] if thumb else 0.0)
        else:
            q.append({"3": p["spatula_middle"], "4": p["spatula_distal"]}.get(joint, 0.0))
    return np.array(q)


def bag_scoop(home, home_rotation, p):
    """Lift the plain bag by its ends from below: each Hand, palm up and fingers pointing at the bag,
    slides its fingers (bent back like a spatula's edge) under an end, into the room its raised underside
    leaves, closes the thumb over the top face, lifts, holds, sets the bag down, lets go and goes Home.
    The left Hand goes first and holds its end before the right one slides in: pushing both in at once
    is an unstable balance that slides the bag towards one Hand."""
    open_hand = {s: np.zeros(len(scene.HAND_JOINTS)) for s in scene.SIDES}
    flat = {s: scoop_hand(p, s, thumb=False) for s in scene.SIDES}
    closed = {s: scoop_hand(p, s) for s in scene.SIDES}
    scoop = {s: spatula(s, np.radians(p["pitch_deg"])) for s in scene.SIDES}

    def at(x, z, y=p["y"]):
        """``x``: the same for both Hands, or {side: x}."""
        x = x if isinstance(x, dict) else {s: x for s in scene.SIDES}
        return {s: np.array([scene.BAG_XY[0] + side_sign(s) * x[s], y, z]) for s in scene.SIDES}

    low, high = p["z"], p["z"] + p["lift"]
    # Pushing both Hands in at once is an unstable balance: the bag slides towards one of them. So the
    # left Hand slides under its end and closes its thumb (holding the bag in place), then the right one.
    left_in = {"left": p["x"], "right": p["x_clear"]}
    left_held = {"left": closed["left"], "right": flat["right"]}
    # The right Hand, sliding in against a bag already held, goes this much deeper to share the load.
    both_in = {"left": p["x"], "right": p["x"] - p["right_deeper"]}
    held = {s: both_in[s] - p["follow"] for s in scene.SIDES}
    return [
        Keyframe(0.0, dict(home), open_hand, dict(home_rotation)),
        Keyframe(1.0, dict(home), open_hand, dict(home_rotation)),
        Keyframe(3.5, at(p["x_clear"], low + 0.06), flat, scoop),            # outside the ends
        Keyframe(5.0, at(p["x_clear"], low), flat, scoop),                   # down to the Table
        Keyframe(7.0, at(left_in, low), flat, scoop),                        # the left fingers slide under
        Keyframe(7.5, at(left_in, low), left_held, scoop),                   # the left thumb holds the end
        Keyframe(9.5, at(both_in, low), left_held, scoop),                   # the right fingers slide under
        Keyframe(10.0, at(both_in, low), closed, scoop),                     # the right thumb closes
        Keyframe(14.0, at(held, high), closed, scoop),                       # lift
        Keyframe(16.0, at(held, high), closed, scoop),                       # hold
        Keyframe(19.5, at(both_in, low + p["set_down"]), closed, scoop),     # set it down
        Keyframe(20.5, at(both_in, low + p["set_down"]), flat, scoop),       # thumbs off
        Keyframe(22.0, at(p["x_clear"], low + p["set_down"]), flat, scoop),  # slide out
        Keyframe(23.5, at(p["x_clear"], low + 0.08), open_hand, scoop),
        Keyframe(26.0, dict(home), open_hand, dict(home_rotation)),
        Keyframe(27.0, dict(home), open_hand, dict(home_rotation)),
    ]


BAG_SCOOP_PARAMETERS = {
    "y": -0.58,            # Hand-base y: the fingers under the middle of the end's width
    "z": 0.06,             # Hand-base height: the fingertips ~4 cm up, by the rounded end's lower curve
    "pitch_deg": 10.0,     # fingers tilted down: the back of the Hand stays clear of the Table
    "x_clear": 0.42,       # Hand-base distance from the bag's centre before sliding in
    "x": 0.32,             # ... with the fingers under the end (fingertips ~5 cm in from it)
    "follow": 0.02,        # how far the Hands close in while lifting [m]
    "right_deeper": 0.02,  # the right Hand (sliding in second) goes this much deeper [m]
    "spatula_middle": -0.2,  # fingers 2-5 bent back: the tips down on the Table, under the end's curve [rad]
    "spatula_distal": -0.3,
    "thumb_1": 1.5,        # left thumb joints _1.._4: its tip over the top face near the end [rad] (right: mirrored)
    "thumb_2": 2.5,
    "thumb_3": 0.0,
    "thumb_4": 0.0,
    "lift": 0.20,          # [m]
    "set_down": 0.005,     # [m]
}
SCRIPTS = {"bag_lift": (bag_lift, BAG_LIFT_PARAMETERS), "bag_lift_plain": (bag_scoop, BAG_SCOOP_PARAMETERS)}


def interpolate(keys, t):
    """Hand-base positions, Hand joints and Hand-base orientations at time ``t`` (minimum jerk)."""
    if t >= keys[-1].time:
        return keys[-1].position, keys[-1].hand_q, keys[-1].rotation
    i = max(k for k in range(len(keys) - 1) if keys[k].time <= t)
    a, b = keys[i], keys[i + 1]
    s = minimum_jerk((t - a.time) / max(b.time - a.time, 1e-9))
    position = {side: (1 - s) * np.asarray(a.position[side]) + s * np.asarray(b.position[side]) for side in scene.SIDES}
    hand_q = {side: (1 - s) * a.hand_q[side] + s * b.hand_q[side] for side in scene.SIDES}
    rotation = {side: Slerp([0.0, 1.0], Rotation.from_matrix([a.rotation[side], b.rotation[side]]))(s).as_matrix()
                for side in scene.SIDES}
    return position, hand_q, rotation


def run(object_name, script, parameters, record=False, video=None, output_dir=DEMONSTRATIONS):
    """Plays the script; returns (trace dict, path of the saved Demonstration or None)."""
    model = scene.build_model(object_name)
    data = mujoco.MjData(model)
    dt = 1.0 / CONTROL_HZ
    output = SimOutput(model, data, dt)
    ik = BimanualIK(dt)                                  # starts at Home, like the Scene (SimOutput resets it)
    home_rotation = {s: ik.hand_base_pose(s).rotation.copy() for s in scene.SIDES}
    home = {s: ik.hand_base_pose(s).position.copy() for s in scene.SIDES}
    keys = SCRIPTS[script][0](home, home_rotation, parameters)

    recorder = None
    if record:
        recorder = DemonstrationRecorder(output_dir, {
            "source": "scripted", "control_hz": CONTROL_HZ,
            "config": {"script": script, "parameters": parameters},
            "scene": {"base_pos": scene.BASE_POS, "home_arm_q": scene.HOME_ARM_Q, **scene.object_metadata(object_name)}})
        recorder.start()
    renderer = frames = None
    if video:
        renderer = mujoco.Renderer(model, 540, 960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (0.0, -0.5, 0.12)
        camera.distance, camera.azimuth, camera.elevation = 1.3, 115.0, -25.0
        frames = []

    trace = {"time": [], "target": [], "measured": [], "object": [], "object_joints": [], "contacts": []}
    steps = int(round(keys[-1].time * CONTROL_HZ)) + 1
    for k in range(steps):
        t = k * dt
        position, hand_q, rotation = interpolate(keys, t)
        targets_pose = {s: Pose(position[s], rotation[s]) for s in scene.SIDES}
        arm_q = ik.step(targets_pose, hand_q)
        targets = {s: LimbTargets(arm_q[s], hand_q[s], targets_pose[s], True) for s in scene.SIDES}
        output.send(targets)
        measured = output.measured()
        if recorder:
            recorder.add(t, Inputs(), targets, measured)
        trace["time"].append(t)
        trace["target"].append({s: position[s].copy() for s in scene.SIDES})
        trace["measured"].append({s: measured[s]["hand_base"][:3].copy() for s in scene.SIDES})
        trace["object"].append(measured["object"].copy())
        trace["object_joints"].append(measured["object_joints"].copy())
        trace["contacts"].append(hand_object_contacts(model, data))
        if renderer is not None and k % 2 == 0:
            renderer.update_scene(data, camera)
            frames.append(renderer.render())
    path = recorder.save(True, "scripted") if recorder else None
    if video:
        write_video(video, frames, fps=CONTROL_HZ / 2)
        renderer.close()
    return trace, path


def write_video(path, frames, fps):
    """H.264 .mp4 through the system ffmpeg (the teleoperation environment has no video library)."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("--video needs ffmpeg on the PATH")
    height, width = frames[0].shape[:2]
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)]
    subprocess.run(command, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames), check=True)


def hand_object_contacts(model, data):
    """{side: total normal force [N] between that Hand and the Object's bodies}."""
    force = {s: 0.0 for s in scene.SIDES}
    wrench = np.zeros(6)
    for i, c in enumerate(data.contact[: data.ncon]):
        names = [model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
        sides = [n.split("/", 1)[0] for n in names]
        is_object = [n == "object" or n.startswith(("bag_", "book_")) for n in names]
        for side in scene.SIDES:
            if side in sides and any(is_object):
                mujoco.mj_contactForce(model, data, i, wrench)
                force[side] += abs(wrench[0])
    return force


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", default="bag_lift", choices=sorted(SCRIPTS))
    parser.add_argument("--object", default="bag_sealed", choices=scene.OBJECTS)
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--video", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEMONSTRATIONS)
    args = parser.parse_args(argv)
    parameters = dict(SCRIPTS[args.script][1])
    for pair in args.param:
        name, value = pair.split("=", 1)
        if name not in parameters:
            raise SystemExit(f"unknown parameter {name!r}; known: {sorted(parameters)}")
        parameters[name] = float(value)
    trace, path = run(args.object, args.script, parameters, args.record, args.video, args.output_dir)
    z = np.array([o[2] for o in trace["object"]])
    print(json.dumps({"object_z_start": float(z[0]), "object_z_max": float(z.max()), "saved": str(path) if path else None}))


if __name__ == "__main__":
    main()
