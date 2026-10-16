"""A rollout next to its Reference, as an .mp4 rendered with the MuJoCo Scene.

Left: the policy's joints as tracked in Isaac Lab, with the simulated Object in
Training stage 2 (stage 1 has none). Right: the Reference with the Object as the Demonstration recorded
it. An articulated Object (the book, the bag) is drawn with its joints (the cover's hinge, the bag's bends and twists)
as simulated and as recorded, and the Scene's extra free bodies (the cube beside
the basin) at their simulated and recorded poses too. Both use the README camera (scripts/render_setup.py). Errors are the worst
Limb at each frame, as scripts/evaluate.py scores them.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from bimanual_dg5fs import scene  # noqa: E402
from bimanual_training import limbs  # noqa: E402

WIDTH, HEIGHT = 960, 540
FPS = 50
HOLD_S = 1.5                    # the last frame is held this long
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def drawn_object(cfg):
    """The Object to draw for an environment cfg: its own (Training stage 2 and 3) or,
    in stage 1, which simulates none, the one its Reference was demonstrated with."""
    from bimanual_training.reference import reference_object

    return getattr(cfg, "object", None) or reference_object(cfg.reference)


def object_mesh(object_name):
    """``(vertices (V, 3), faces (F, 3))`` of the Object in its body frame, from the
    MuJoCo Scene's geoms (an articulated Object at its start pose, the book closed)."""
    import trimesh

    model = scene.build_model(object_name)
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    root = model.body("object").id
    origin, rotation = data.xpos[root], data.xmat[root].reshape(3, 3)
    parts = []
    for g in range(model.ngeom):
        body = model.geom_bodyid[g]
        while body not in (root, 0):
            body = model.body_parentid[body]
        if body != root:
            continue
        size, kind = model.geom_size[g], model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            part = trimesh.creation.box(extents=2.0 * size)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            part = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1], sections=32)
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            part = trimesh.creation.capsule(height=2.0 * size[1], radius=size[0], count=(8, 8))
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            faces = model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
            part = trimesh.Trimesh(vertices=model.mesh_vert[start:start + count], faces=faces, process=False)
        else:
            continue
        transform = np.eye(4)
        transform[:3, :3] = rotation.T @ data.geom_xmat[g].reshape(3, 3)
        transform[:3, 3] = rotation.T @ (data.geom_xpos[g] - origin)
        part.apply_transform(transform)
        parts.append(part)
    mesh = trimesh.util.concatenate(parts)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def joint_caption(object_name, joint_q):
    """A caption line for an articulated Object's joints (scene.OBJECT_JOINTS order)."""
    if object_name in ("bag", "bag_sealed"):
        names = scene.OBJECT_JOINTS[object_name]
        bends = [abs(q) for name, q in zip(names, joint_q) if "_bend_" in name]
        twists = [abs(q) for name, q in zip(names, joint_q) if "_twist_" in name]
        return f"bend max {np.degrees(max(bends)):.0f} deg, twist max {np.degrees(max(twists)):.0f} deg"
    return f"cover {np.degrees(joint_q[0]):.0f} deg"


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


class SceneRenderer:
    def __init__(self, object_name="cube"):
        self.model = scene.build_model(object_name)
        self.data = mujoco.MjData(self.model)
        scene.reset_to_home(self.model, self.data)
        self.joints = [np.array([self.model.joint(f"{side}/{j}").qposadr[0] for j in limbs.JOINTS])
                       for side in limbs.SIDES]
        self.object = self.model.joint("object_free").qposadr[0]
        self.object_joints = np.array([self.model.joint(name).qposadr[0]
                                       for name in scene.OBJECT_JOINTS[object_name]], dtype=int)
        # The extra free bodies (scene.EXTRA_OBJECTS), each on its "<name>_free" joint.
        self.extras = {name: self.model.joint(f"{name}_free").qposadr[0]
                       for name in scene.EXTRA_OBJECTS.get(object_name, ())}
        self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = (0.0, -0.28, 0.30)
        self.camera.distance = 1.9
        self.camera.azimuth = 125.0
        self.camera.elevation = -30.0

    def render(self, q, object_pose=None, object_joint_q=None, extra_poses=None):
        """``object_joint_q``: an articulated Object's joints (scene.OBJECT_JOINTS order); None draws them at 0.
        ``extra_poses``: {name: xyz + xyzw} of the extra free bodies; a missing one is hidden, as the
        Object is when ``object_pose`` is None."""
        for limb, addresses in enumerate(self.joints):
            self.data.qpos[addresses] = q[limb]
        self.data.qpos[self.object_joints] = 0.0 if object_joint_q is None else object_joint_q
        extra_poses = extra_poses or {}
        for address, pose in [(self.object, object_pose)] + [(a, extra_poses.get(n)) for n, a in self.extras.items()]:
            if pose is None:
                self.data.qpos[address:address + 7] = [0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0]
            else:
                x, y, z, qx, qy, qz, qw = pose
                self.data.qpos[address:address + 7] = [x, y, z, qw, qx, qy, qz]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def _caption(image, lines, color=(255, 255, 255)):
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    bold, regular = _font(FONT_BOLD, 24), _font(FONT, 20)
    y = 12
    for i, line in enumerate(lines):
        draw.text((16, y), line, font=bold if i == 0 else regular, fill=color if i == 0 else (230, 230, 230),
                  stroke_width=2, stroke_fill=(0, 0, 0))
        y += 30
    return np.asarray(canvas)


def _banner(image, text, color):
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, canvas.height - 56, canvas.width, canvas.height), fill=(0, 0, 0))
    draw.text((16, canvas.height - 44), text, font=_font(FONT_BOLD, 26), fill=color)
    return np.asarray(canvas)


def write_rollout_video(path, trace, reference, label, result, object_name="cube"):
    """``trace`` from scripts/evaluate.py's rollout(); ``reference`` a bimanual_training Reference;
    ``object_name`` the Object drawn (bimanual_dg5fs.scene.OBJECTS)."""
    import imageio

    reference_q = reference.q.cpu().numpy()
    object_pose = reference.object_pose.cpu().numpy()
    last = reference.last_index
    renderer = SceneRenderer(object_name)
    if result["terminated_at"] is not None:
        verdict = f"early termination at frame {result['terminated_at']}"
    else:
        verdict = "SUCCESS" if result["success"] else "reached the end"
    summary = (f"{verdict}: palm {100 * result['palm_mean_m']:.2f} cm, "
               f"hand {result['hand_rms_mean_rad']:.3f} rad (means)")
    if "object_mean_m" in result:
        summary += f", object {100 * result['object_mean_m']:.1f} cm"
    if "cube_mean_m" in result:
        summary += (f", cube {100 * result['cube_mean_m']:.1f} cm, "
                    f"{'in' if result['cube_final_in_basin'] else 'not in'} the basin")
    if "object_joint_error_mean_rad" in result:
        summary += (f", {'joints' if reference.object_joints > 1 else 'hinge'} "
                    f"{np.degrees(result['object_joint_error_mean_rad']):.0f} deg")
    reference_joints = reference.object_joint_q.cpu().numpy()
    joints = trace.get("object_joint_q")
    # The extra free bodies: simulated (left; none in stage 1) and as recorded (right).
    extra_names = scene.EXTRA_OBJECTS.get(object_name, ())
    simulated_extras = {name: trace.get(f"{name}_pose") for name in extra_names}
    recorded_extras = {name: (trace.get(f"reference_{name}") if trace.get(f"reference_{name}") is not None
                              else getattr(reference, f"{name}_pose", None)) for name in extra_names}
    recorded_extras = {name: track.cpu().numpy() if hasattr(track, "cpu") else track
                       for name, track in recorded_extras.items()}
    recorded_joints = trace.get("reference_object_joint_q")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=7, macro_block_size=4) as writer:
        frame = None
        objects = trace.get("object_pose")
        for k, (q, index, palm, hand) in enumerate(zip(trace["q"], trace["reference_index"], trace["palm_m"],
                                                       trace["hand_rad"])):
            joint_q = None if joints is None else joints[k]
            right_joint_q = recorded_joints[k] if recorded_joints is not None else reference_joints[index]
            left_lines = [f"policy: {label}", f"frame {index}/{last}   palm {100 * palm:.1f} cm   hand {hand:.3f} rad"]
            right_lines = ["Reference", f"start frame {result['start']}"]
            if joint_q is not None and len(joint_q):
                left_lines.append(joint_caption(object_name, joint_q))
            if len(right_joint_q):
                right_lines.append(joint_caption(object_name, right_joint_q))
            left_extras = {name: track[k] for name, track in simulated_extras.items() if track is not None}
            right_extras = {name: track[k] if trace.get(f"reference_{name}") is not None else track[index]
                            for name, track in recorded_extras.items() if track is not None}
            left = _caption(renderer.render(q, None if objects is None else objects[k], joint_q, left_extras),
                            left_lines)
            right_q = trace["reference_q"][k] if trace.get("reference_q") is not None else reference_q[index]
            right_object = (trace["reference_object"][k] if trace.get("reference_object") is not None
                            else object_pose[index])
            right = _caption(renderer.render(right_q, right_object, right_joint_q if len(right_joint_q) else None,
                                             right_extras),
                             right_lines + ([trace["placement"]] if trace.get("placement") else []),
                             color=(120, 230, 140))
            frame = np.concatenate((left, right), axis=1)
            writer.append_data(frame)
        if frame is not None:
            final = _banner(frame, summary, (255, 110, 110) if result["terminated_at"] is not None
                            else (120, 230, 140) if result["success"] else (255, 255, 255))
            for _ in range(int(HOLD_S * FPS)):
                writer.append_data(final)
    renderer.close()
