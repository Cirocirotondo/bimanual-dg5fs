#!/usr/bin/env bash
# Evaluates a wave's runs on the desktop GPU (videos included) and writes the wave
# table. Run from bimanual-dg5fs/ after the runs are pulled (wait_wave.sh does both).
#
#   training/fleet/evaluate_wave.sh M1 training/logs/motion_imitation/<run> [<run> ...]
#
# Runs already evaluated at their newest checkpoint are not evaluated again.
# Per run: <run>/eval/<checkpoint>/{start_<N>.json,start_<N>.mp4,summary.md}
# (training/scripts/evaluate_run.py). Wave table: training/logs/waves/<wave>.md,
# one line per run from frame 0, then every run's own summary.
set -euo pipefail
cd "$(dirname "$0")/../.."
wave="${1:?wave name}"; shift
[ $# -gt 0 ] || { echo "no run directories" >&2; exit 2; }
# Only runs whose newest checkpoint has no evaluation yet.
todo=()
for run in "$@"; do
  newest=$(ls "$run"/model_*.pt 2>/dev/null | sed -n 's/.*model_\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
  [ -n "$newest" ] || { echo "no checkpoint in $run" >&2; continue; }
  [ -f "$run/eval/model_${newest}/summary.md" ] || todo+=("$run")
done
[ ${#todo[@]} -eq 0 ] || env -u PYTHONPATH deps/IsaacLab/.venv/bin/python training/scripts/evaluate_run.py "${todo[@]}"
mkdir -p training/logs/waves
env -u PYTHONPATH deps/IsaacLab/.venv/bin/python - "$wave" "$@" <<'EOF'
import json, sys
from datetime import datetime
from pathlib import Path

wave, runs = sys.argv[1], [Path(r) for r in sys.argv[2:]]
table = Path("training/logs/waves") / f"{wave}.md"
lines = [f"# Wave {wave}: evaluation", "",
         f"Written {datetime.now():%Y-%m-%d %H:%M} by training/fleet/evaluate_wave.sh. Deterministic rollouts,",
         "5 repeats per start frame (median [min - max]); success = whole Reference, palm < 2 cm,",
         "hand RMS < 0.1 rad. Videos: policy (left) next to the Reference (right).", "",
         "Smoothness (frame 0): RMS acceleration of the drive targets over the Reference's (x1 = as smooth as the demo).", "",
         "| run | checkpoint | from frame 0: successes | early term. | palm [cm] | hand RMS [rad] | arm / hand accel. xRef | later starts: successes | video (frame 0) |",
         "|---|---|---|---|---|---|---|---|---|"]
details = []
for run in runs:
    evals = [p for p in (run / "eval").iterdir() if (p / "summary.md").is_file()] if (run / "eval").is_dir() else []
    if not evals:
        continue
    out = max(evals, key=lambda p: int(p.name.split("_")[1]))
    starts = sorted((json.loads(p.read_text()) for p in out.glob("start_*.json")), key=lambda a: a["start"])
    first, later = starts[0], starts[1:]
    later_ok = sum(a["successes"] for a in later), sum(a["repeats"] for a in later)
    rel = Path("..") / "motion_imitation" / run.name / "eval" / out.name
    lines.append(
        f"| {run.name} | {out.name} | {first['successes']}/{first['repeats']} | "
        f"{first['terminations']}/{first['repeats']} | {first['palm_mean_cm'][0]:.2f} "
        f"[{first['palm_mean_cm'][1]:.2f} - {first['palm_mean_cm'][2]:.2f}] | "
        f"{first['hand_rms_mean_rad'][0]:.3f} | {first.get('arm_target_acc_ratio', float('nan')):.2f} / "
        f"{first.get('hand_target_acc_ratio', float('nan')):.2f} | {later_ok[0]}/{later_ok[1]} | "
        f"[start_{first['start']}.mp4]({rel}/start_{first['start']}.mp4) |")
    if first.get("object_mean_cm"):
        lines.append(f"|  | Object from frame 0: mean {first['object_mean_cm'][0]:.1f} cm, max "
                     f"{first['object_max_cm'][0]:.1f} cm, lifted {first['object_lift_cm'][0]:.1f} cm |  |  |  |  |  |  |  |")
    details.append((out / "summary.md").read_text().replace("(start_", f"({rel}/start_"))
table.write_text("\n".join(lines) + "\n\n" + "\n".join(details))
print(table)
EOF
