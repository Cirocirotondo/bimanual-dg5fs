#!/usr/bin/env python3
"""Open the Scene in the MuJoCo viewer at the Home pose.

Every Limb joint has a position actuator: move them with the sliders in the
viewer's Control panel (right side). Backspace resets to the Home keyframe.

    uv run scripts/view_scene.py
    uv run scripts/view_scene.py --screenshot scene.png   # render one frame, no window
"""
import argparse

import mujoco
import mujoco.viewer

from bimanual_dg5fs import scene


def set_camera(cam):
    # From the operator's side (behind the Arm bases, looking along -y), so the
    # left Limb appears on the left.
    cam.lookat[:] = (-0.08, -0.45, 0.20)
    cam.distance = 2.0
    cam.azimuth = -60.0
    cam.elevation = -35.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", help="render the Home pose to this PNG and exit")
    args = parser.parse_args()

    model = scene.build_model()
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)

    if args.screenshot:
        from PIL import Image

        cam = mujoco.MjvCamera()
        set_camera(cam)
        with mujoco.Renderer(model, height=720, width=1280) as renderer:
            renderer.update_scene(data, camera=cam)
            Image.fromarray(renderer.render()).save(args.screenshot)
        print(f"wrote {args.screenshot}")
        return

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
