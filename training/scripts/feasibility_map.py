#!/usr/bin/env python3
"""Feasibility map (CONTEXT.md): which Object placements the retargeted Reference can execute.

For a regular grid of placements (x, y along the Table, yaw folded into ±45°) both
Arms are re-solved by IK in the Demonstration's joint family (retarget.py) and each
placement is checked over the whole clip, in this order; it records the first
criterion it fails:

  ik          the Hand base misses its target by more than 1 mm or 10 mrad
  limit       an Arm joint comes within 0.05 rad of its limit
  speed       an Arm joint moves faster than 0.5 * pi rad/s
  collision   the two Limbs touch each other, or a Limb touches the Table or floor
              (MuJoCo Scene in kinematic mode: joints set, contacts read, no dynamics;
              contacts with the Object and within one Limb do not count)

(simtoolreal_newton checked the first three with the same thresholds; it had no
collision check. Singularities are not checked yet.)

Writes training/logs/feasibility/<name>/: map.npz (grid, first failure, metrics),
map.html (one map per yaw, the largest box with >= 90% feasible placements that
contains the Demonstration's placement), and prints the box.

    env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/feasibility_map.py \
        [--reference lift_cube_2] [--extent 0.25] [--step 0.025] [--yaw 45] [--yaw-step 7.5]
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from bimanual_training import LOGS_DIR, REFERENCES_DIR, limbs
from bimanual_training.retarget import PlacementRetargeter

POSITION_TOLERANCE_M = 1e-3
ROTATION_TOLERANCE_RAD = 10e-3
LIMIT_MARGIN_RAD = 0.05
SPEED_LIMIT_RAD_S = 0.5 * math.pi
# Smallest singular value of either Arm's Hand-base Jacobian over the clip. The demo
# stays above 0.15; placements whose Arm drops to 0.02-0.05 failed every Generalised
# imitation run (the policy's damped IK step, lambda 0.02, loses up to half a command).
SINGULARITY_SIGMA = 0.10
REASONS = ("feasible", "ik", "limit", "speed", "singular", "collision")
FEASIBLE_FRACTION = 0.9

# ----------------------------------------------------------------- collisions

_WORKER = {}


def _collision_worker_init():
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from bimanual_dg5fs import scene

    model = scene.build_model()
    data = mujoco.MjData(model)
    scene.reset_to_home(model, data)
    body_class = []
    for b in range(model.nbody):
        name = model.body(b).name
        side = name.split("/", 1)[0]
        body_class.append(side if side in limbs.SIDES else ("table" if name in ("table", "world") else name))
    geom_class = [body_class[model.geom_bodyid[g]] for g in range(model.ngeom)]
    addresses = [np.array([model.joint(f"{side}/{j}").qposadr[0] for j in limbs.JOINTS]) for side in limbs.SIDES]
    _WORKER.update(mujoco=mujoco, model=model, data=data, geom_class=geom_class, addresses=addresses,
                   object_adr=model.joint("object_free").qposadr[0])


def _collides(args):
    """First colliding frame index of one placement's clip, or -1."""
    q, objects = args                     # q (k, 2, 26), objects (k, 7) xyz + xyzw
    w = _WORKER
    mujoco, model, data, classes = w["mujoco"], w["model"], w["data"], w["geom_class"]
    for k in range(len(q)):
        for limb, address in enumerate(w["addresses"]):
            data.qpos[address] = q[k, limb]
        x, y, z, qx, qy, qz, qw = objects[k]
        data.qpos[w["object_adr"]:w["object_adr"] + 7] = (x, y, z, qw, qx, qy, qz)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        for c in data.contact[: data.ncon]:
            a, b = classes[c.geom1], classes[c.geom2]
            if c.dist >= 0.0 or "object" in (a, b) or a == b:
                continue
            if a in limbs.SIDES or b in limbs.SIDES:      # Limb-Limb, Limb-Table, Limb-floor
                return k
    return -1


def check_placements(retargeter, translation, yaw, frame_stride, pool, keep_q=False,
                     singularity_sigma=SINGULARITY_SIGMA):
    """Retarget a batch of placements and check them; reason codes as in REASONS.

    Collisions are checked only where every other criterion holds. With ``keep_q``
    the Arms' joints (P, k, 2, 6) come back too.
    """
    dt = 0.02 * frame_stride
    frames, results = retargeter.solve(translation, yaw, frame_stride=frame_stride)
    position = torch.stack([r["position_residual"].amax(1) for r in results]).amax(0)
    rotation = torch.stack([r["rotation_residual"].amax(1) for r in results]).amax(0)
    margin = torch.stack([torch.minimum(r["arm_q"] - r["lower"], r["upper"] - r["arm_q"]).amin((1, 2))
                          for r in results]).amin(0)
    speed = torch.stack([(r["arm_q"].diff(dim=1).abs() / dt).amax((1, 2)) for r in results]).amax(0)
    sigma = torch.stack([r["sigma_min"].amin(1) for r in results]).amin(0)
    bad_ik = (position > POSITION_TOLERANCE_M) | (rotation > ROTATION_TOLERANCE_RAD)
    bad_limit = ~bad_ik & (margin < LIMIT_MARGIN_RAD)
    bad_speed = ~bad_ik & ~bad_limit & (speed > SPEED_LIMIT_RAD_S)
    bad_singular = ~bad_ik & ~bad_limit & ~bad_speed & (sigma < singularity_sigma)
    reason = np.zeros(len(yaw), dtype=np.int8)
    reason[bad_ik.cpu().numpy()] = 1
    reason[bad_limit.cpu().numpy()] = 2
    reason[bad_speed.cpu().numpy()] = 3
    reason[bad_singular.cpu().numpy()] = 4
    arm = torch.stack([res["arm_q"] for res in results], dim=2).cpu().numpy()                 # (P, k, 2, 6)
    first_collision = {}
    candidates = np.flatnonzero(reason == 0)
    if len(candidates):
        hand = retargeter.hand_q(frames).cpu().numpy()                                         # (k, 2, 20)
        objects = retargeter.object_track(translation.to(retargeter.device, retargeter.dtype),
                                          yaw.to(retargeter.device, retargeter.dtype), frames).cpu().numpy()
        # Collisions on every second frame at most (the map's resolution).
        sub = max(1, 2 // frame_stride)
        jobs = [(np.concatenate((arm[c, ::sub], hand[::sub]), axis=-1), objects[c, ::sub]) for c in candidates]
        hits = pool.map(_collides, jobs, chunksize=4)
        for c, hit in zip(candidates, hits):
            if hit >= 0:
                reason[c] = 5
                first_collision[int(c)] = int(frames[hit * sub])
    out = {"reason": reason, "position": position.cpu().numpy(), "rotation": rotation.cpu().numpy(),
           "sigma_min": sigma.cpu().numpy(),
           "margin": margin.cpu().numpy(), "speed": speed.cpu().numpy(), "first_collision_frame": first_collision,
           "frames": frames.cpu().numpy()}
    if keep_q:
        out["arm_q"] = arm
    return out


# ------------------------------------------------------------------- the box

def largest_box(feasible, xs, ys, yaws, demo_index, minimum=FEASIBLE_FRACTION):
    """Largest axis-aligned box (x, y, yaw index ranges) whose feasible fraction is
    >= ``minimum`` and that contains the Demonstration's placement; size = number of
    grid placements inside."""
    f = feasible.astype(np.int64)                        # (X, Y, A)
    cumulative = np.zeros((f.shape[0] + 1, f.shape[1] + 1, f.shape[2] + 1), dtype=np.int64)
    cumulative[1:, 1:, 1:] = f.cumsum(0).cumsum(1).cumsum(2)
    ix0, iy0, ia0 = demo_index
    best = None
    X, Y, A = f.shape
    x_lo, x_hi = np.arange(0, ix0 + 1), np.arange(ix0 + 1, X + 1)
    for a_lo in range(0, ia0 + 1):
        for a_hi in range(ia0 + 1, A + 1):
            for y_lo in range(0, iy0 + 1):
                for y_hi in range(iy0 + 1, Y + 1):
                    # all x ranges at once
                    xl, xh = np.meshgrid(x_lo, x_hi, indexing="ij")
                    count = (cumulative[xh, y_hi, a_hi] - cumulative[xl, y_hi, a_hi] - cumulative[xh, y_lo, a_hi]
                             - cumulative[xh, y_hi, a_lo] + cumulative[xl, y_lo, a_hi] + cumulative[xl, y_hi, a_lo]
                             + cumulative[xh, y_lo, a_lo] - cumulative[xl, y_lo, a_lo])
                    size = (xh - xl) * (y_hi - y_lo) * (a_hi - a_lo)
                    ok = count >= minimum * size
                    if not ok.any():
                        continue
                    k = np.argmax(np.where(ok, size, -1))
                    i, j = np.unravel_index(k, size.shape)
                    if best is None or size[i, j] > best[0]:
                        best = (int(size[i, j]), float(count[i, j] / size[i, j]),
                                (int(xl[i, j]), int(xh[i, j]) - 1), (y_lo, y_hi - 1), (a_lo, a_hi - 1))
    if best is None:
        return None
    size, fraction, (x0, x1), (y0, y1), (a0, a1) = best
    return {"placements": size, "feasible_fraction": fraction,
            "x_m": [float(xs[x0]), float(xs[x1])], "y_m": [float(ys[y0]), float(ys[y1])],
            "yaw_deg": [float(np.degrees(yaws[a0])), float(np.degrees(yaws[a1]))]}


# -------------------------------------------------------------------- report

COLORS = {"feasible": "var(--ok)", "ik": "var(--ik)", "limit": "var(--limit)",
          "speed": "var(--speed)", "singular": "var(--singular)", "collision": "var(--collision)"}
LABELS = {"feasible": "fattibile", "ik": "IK non converge", "limit": "limite dei giunti",
          "speed": "velocità dei giunti", "singular": "vicino a una singolarità", "collision": "collisione"}


def write_html(path, xs, ys, yaws, reason, box, summary):
    cell = 14
    maps = []
    for a, yaw in enumerate(yaws):
        rects = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                r = REASONS[reason[i, j, a]]
                # x to the right, y upwards (y grows away from the Operator)
                rects.append(f'<rect x="{i * cell}" y="{(len(ys) - 1 - j) * cell}" width="{cell - 1}" '
                             f'height="{cell - 1}" fill="{COLORS[r]}"><title>x {x * 100:+.1f} cm, y {y * 100:+.1f} cm: '
                             f'{LABELS[r]}</title></rect>')
        if box and box["yaw_deg"][0] - 1e-6 <= np.degrees(yaw) <= box["yaw_deg"][1] + 1e-6:
            bx0 = int(round((box["x_m"][0] - xs[0]) / (xs[1] - xs[0])))
            bx1 = int(round((box["x_m"][1] - xs[0]) / (xs[1] - xs[0])))
            by0 = int(round((box["y_m"][0] - ys[0]) / (ys[1] - ys[0])))
            by1 = int(round((box["y_m"][1] - ys[0]) / (ys[1] - ys[0])))
            rects.append(f'<rect class="box" x="{bx0 * cell - 1}" y="{(len(ys) - 1 - by1) * cell - 1}" '
                         f'width="{(bx1 - bx0 + 1) * cell + 1}" height="{(by1 - by0 + 1) * cell + 1}"/>')
        centre = len(xs) // 2
        rects.append(f'<circle cx="{centre * cell + cell / 2 - 0.5}" cy="{(len(ys) - 1 - len(ys) // 2) * cell + cell / 2 - 0.5}" r="3" class="demo"/>')
        share = (reason[:, :, a] == 0).mean()
        maps.append(f'<figure><figcaption>yaw {np.degrees(yaw):+.1f}&deg; &middot; {share:.0%} fattibili</figcaption>'
                    f'<svg viewBox="-1 -1 {len(xs) * cell + 2} {len(ys) * cell + 2}" role="img" '
                    f'aria-label="Mappa di fattibilità a yaw {np.degrees(yaw):.1f} gradi">{"".join(rects)}</svg></figure>')
    legend = "".join(f'<li><span style="background:{COLORS[r]}"></span>{LABELS[r]} '
                     f'<b>{summary["counts"][r]}</b></li>' for r in REASONS)
    box_text = ("nessun box con il 90% di pose fattibili contiene la posa della demo" if box is None else
                f'x {box["x_m"][0] * 100:+.1f}…{box["x_m"][1] * 100:+.1f} cm, y {box["y_m"][0] * 100:+.1f}…'
                f'{box["y_m"][1] * 100:+.1f} cm, yaw {box["yaw_deg"][0]:+.1f}…{box["yaw_deg"][1]:+.1f}&deg; '
                f'({box["feasible_fraction"]:.0%} fattibili, {box["placements"]} pose della griglia)')
    path.write_text(TEMPLATE.replace("{{MAPS}}", "\n".join(maps)).replace("{{LEGEND}}", legend)
                    .replace("{{BOX}}", box_text).replace("{{SUMMARY}}", json.dumps(summary["settings"]))
                    .replace("{{TOTAL}}", f'{summary["counts"]["feasible"]} / {reason.size}'))


TEMPLATE = """<title>Fattibilità lift_cube_2</title>
<style>
:root { --bg:#f3f5f4; --fg:#1d2427; --muted:#5f6a6e; --line:#d4dadb; --surface:#fff;
  --ok:#3f9b63; --ik:#8a8f93; --limit:#d9a21b; --speed:#d1621f; --singular:#7b4fb3; --collision:#b8312a; --box:#1d2427;
  --font: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#121719; --fg:#e5eaeb; --muted:#9aa5a9;
  --line:#2c3539; --surface:#1a2124; --ok:#4fb878; --ik:#6c7378; --limit:#e0b13a; --speed:#ec7b3a; --singular:#a98be0; --collision:#e2574c;
  --box:#e5eaeb; color-scheme: dark; } }
:root[data-theme="dark"] { --bg:#121719; --fg:#e5eaeb; --muted:#9aa5a9; --line:#2c3539; --surface:#1a2124;
  --ok:#4fb878; --ik:#6c7378; --limit:#e0b13a; --speed:#ec7b3a; --singular:#a98be0; --collision:#e2574c; --box:#e5eaeb; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font-family: var(--font); font-size: 15px; line-height: 1.5; }
.page { max-width: 1120px; margin: 0 auto; padding-inline: 20px; padding-block: 32px 56px; display: grid; gap: 24px; }
h1 { margin: 0; font-size: 2rem; text-wrap: balance; }
p { margin: 0; max-width: 72ch; }
.muted { color: var(--muted); }
ul.legend { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 8px 20px; }
ul.legend span { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.maps { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px; display: grid; gap: 6px; min-width: 0; }
figcaption { font-family: var(--mono); font-size: 0.82rem; }
svg { width: 100%; height: auto; display: block; }
rect.box { fill: none; stroke: var(--box); stroke-width: 2; }
circle.demo { fill: var(--surface); stroke: var(--fg); stroke-width: 1.5; }
code { font-family: var(--mono); font-size: 0.82rem; overflow-wrap: anywhere; }
</style>
<div class="page">
  <h1>Fattibilità delle pose del cubo</h1>
  <p>Ogni quadrato è una posa del cubo sul tavolo, spostata rispetto alla demo <code>lift_cube_2</code>. Su ogni pannello, x cresce verso destra e y verso l'alto. Il cerchio è la posa della demo. Il colore indica il primo criterio che la traiettoria ricalcolata con l'IK non rispetta. Il riquadro scuro è il box più grande con almeno il 90% di pose fattibili che contiene la demo.</p>
  <p><b>Pose fattibili:</b> {{TOTAL}} &middot; <b>Box di training:</b> {{BOX}}</p>
  <ul class="legend">{{LEGEND}}</ul>
  <div class="maps">
{{MAPS}}
  </div>
  <p class="muted">Criteri: IK entro 1 mm e 10 mrad, giunti a più di 0.05 rad dai limiti, velocità dei giunti sotto 0.5·π rad/s, valore singolare minimo dello Jacobiano del palmo di ciascun braccio sopra la soglia, nessun contatto tra i due Limb o tra un Limb e il tavolo. Impostazioni: <code>{{SUMMARY}}</code></p>
</div>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default="lift_cube_2")
    parser.add_argument("--extent", type=float, default=0.25, help="half range of x and y [m]")
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--yaw", type=float, default=45.0, help="half range of the yaw [deg]")
    parser.add_argument("--yaw-step", type=float, default=7.5)
    parser.add_argument("--frame-stride", type=int, default=2, help="2 matches the full clip (simtoolreal_newton)")
    parser.add_argument("--batch", type=int, default=2048, help="placements solved together")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--name", default=None)
    parser.add_argument("--singularity-sigma", type=float, default=SINGULARITY_SIGMA,
                        help="minimum singular value of each Arm's Hand-base Jacobian (0 disables)")
    args = parser.parse_args()

    n = int(round(args.extent / args.step))
    xs = np.linspace(-n * args.step, n * args.step, 2 * n + 1)
    ys = xs.copy()
    m = int(round(args.yaw / args.yaw_step))
    yaws = np.radians(np.linspace(-m * args.yaw_step, m * args.yaw_step, 2 * m + 1))
    grid = np.stack(np.meshgrid(xs, ys, yaws, indexing="ij"), axis=-1).reshape(-1, 3)
    out_dir = LOGS_DIR / "feasibility" / (args.name or args.reference)
    out_dir.mkdir(parents=True, exist_ok=True)

    retargeter = PlacementRetargeter(REFERENCES_DIR / f"{args.reference}.npz", device=args.device)
    count = len(grid)
    reason = np.zeros(count, dtype=np.int8)
    metrics = {k: np.zeros(count) for k in ("position_residual_m", "rotation_residual_rad", "limit_margin_rad",
                                            "peak_speed_rad_s", "sigma_min")}
    first_collision_frame = np.full(count, -1)
    started = time.time()
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_collision_worker_init)
    for start in range(0, count, args.batch):
        part = slice(start, min(start + args.batch, count))
        translation = torch.as_tensor(np.c_[grid[part, :2], np.zeros(part.stop - part.start)])
        yaw = torch.as_tensor(grid[part, 2])
        out = check_placements(retargeter, translation, yaw, args.frame_stride, pool,
                               singularity_sigma=args.singularity_sigma)
        r = out["reason"]
        position, rotation, margin, speed = (out["position"], out["rotation"], out["margin"], out["speed"])
        for c, frame in out["first_collision_frame"].items():
            first_collision_frame[start + c] = frame
        reason[part] = r
        metrics["position_residual_m"][part] = position
        metrics["rotation_residual_rad"][part] = rotation
        metrics["limit_margin_rad"][part] = margin
        metrics["peak_speed_rad_s"][part] = speed
        metrics["sigma_min"][part] = out["sigma_min"]
        done = part.stop
        print(f"{done}/{count} placements, {time.time() - started:.0f} s, feasible so far "
              f"{(reason[:done] == 0).mean():.0%}", flush=True)
    pool.close()

    shape = (len(xs), len(ys), len(yaws))
    reason3 = reason.reshape(shape)
    demo_index = (len(xs) // 2, len(ys) // 2, len(yaws) // 2)
    box = largest_box(reason3 == 0, xs, ys, yaws, demo_index)
    summary = {
        "counts": {name: int((reason == i).sum()) for i, name in enumerate(REASONS)},
        "per_yaw_feasible": {f"{np.degrees(a):+.1f}": float((reason3[:, :, k] == 0).mean()) for k, a in enumerate(yaws)},
        "box": box,
        "settings": {"reference": args.reference, "extent_m": args.extent, "step_m": args.step, "yaw_deg": args.yaw,
                     "yaw_step_deg": args.yaw_step, "frame_stride": args.frame_stride,
                     "position_tolerance_m": POSITION_TOLERANCE_M, "rotation_tolerance_rad": ROTATION_TOLERANCE_RAD,
                     "limit_margin_rad": LIMIT_MARGIN_RAD, "speed_limit_rad_s": SPEED_LIMIT_RAD_S,
                     "singularity_sigma": args.singularity_sigma},
    }
    np.savez_compressed(out_dir / "map.npz", xs=xs, ys=ys, yaws=yaws, reason=reason3,
                        first_collision_frame=first_collision_frame.reshape(shape),
                        **{k: v.reshape(shape) for k, v in metrics.items()})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_html(out_dir / "map.html", xs, ys, yaws, reason3, box, summary)
    print(json.dumps({k: summary[k] for k in ("counts", "per_yaw_feasible", "box")}, indent=2))
    print(f"written {out_dir}")


if __name__ == "__main__":
    main()
