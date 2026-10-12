#!/usr/bin/env python3
"""Renders the Scene at Home for the README (docs/images/bimanual_setup.png).

    MUJOCO_GL=egl env -u PYTHONPATH uv run python scripts/render_setup.py
"""
from pathlib import Path

import mujoco
from PIL import Image

from bimanual_dg5fs import scene

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "bimanual_setup.png"


def main():
    model = scene.build_model()
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.0, -0.28, 0.30)
    camera.distance = 1.9
    camera.azimuth = 125.0
    camera.elevation = -30.0
    renderer = mujoco.Renderer(model, 900, 1600)
    renderer.update_scene(data, camera)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(renderer.render()).save(OUTPUT, optimize=True)
    renderer.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
