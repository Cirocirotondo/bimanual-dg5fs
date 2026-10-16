#!/usr/bin/env bash
# Run one training to an absolute iteration target, restarting it from its
# newest checkpoint whenever the process dies before getting there.
# (Port of simtoolreal_newton's scripts/supervise_train.sh.)
#
#   training/fleet/supervise_train.sh --run-name NAME --target ITER \
#       [--seed-checkpoint PATH] [--python BIN] [--max-attempts K] -- <train.py arguments>
#
# The same script drives a run on the desktop (--python deps/IsaacLab/.venv/bin/python)
# and inside the tars/case container (default python = the image's venv).
#
# Iterations: RSL-RL numbers them from 0 and a resumed run continues from the
# checkpoint's own number, so a run with target T ends with model_<T-1>.pt and
# each attempt passes --max-iterations T - DONE (relative to the resume point).
# A seed checkpoint (e.g. another run's model_2999.pt) is the starting point of
# the first attempt; its iteration is read from its file name.
#
# Files it leaves behind, for whoever is watching (paths from bimanual-dg5fs/):
#   training/logs/queue/<NAME>.log      the [sup] state lines, then the trainer output
#   training/logs/queue/<NAME>.rundir   the run directory, one line, written at start
#   <run dir>/SUPERVISOR_DONE           written when the loop ends, with the outcome
set -u

PY=python
MAX_ATTEMPTS=15
ABORT_LIMIT=3
SEED_CKPT=""
RUN_NAME=""
TARGET=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --seed-checkpoint) SEED_CKPT="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$RUN_NAME" ] && [ -n "$TARGET" ] || { sed -n '2,10p' "$0" >&2; exit 2; }

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
# Only this project's packages: ROS leaves a PYTHONPATH behind on the desktop,
# and in the container the packages are not installed, only mounted.
export PYTHONPATH="$REPO/src:$REPO/training"
LAUNCH_TS=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR="$REPO/training/logs/motion_imitation/${LAUNCH_TS}_${RUN_NAME}"
LOG="$REPO/training/logs/queue/${RUN_NAME}.log"
mkdir -p "$RUN_DIR" "$(dirname "$LOG")"
echo "$RUN_DIR" > "$REPO/training/logs/queue/${RUN_NAME}.rundir"

say() { echo "[sup] $(date +%H:%M:%S) $*" >> "$LOG"; }
iteration_of() { basename "$1" | sed 's/model_\([0-9]*\)\.pt/\1/'; }
say "run dir $RUN_DIR, target $TARGET, seed ${SEED_CKPT:-scratch}"

if [ -n "$SEED_CKPT" ] && [ ! -f "$SEED_CKPT" ]; then
  say "seed checkpoint missing: $SEED_CKPT"
  echo "status=failed reason=missing_seed" > "$RUN_DIR/SUPERVISOR_DONE"
  exit 1
fi

# Newest model_<N>.pt in the run directory, or nothing before the first save.
latest_ckpt() {
  ls "$RUN_DIR"/model_*.pt 2>/dev/null \
    | sed -n 's/.*model_\([0-9]*\)\.pt$/\1 &/p' | sort -n | tail -1 | cut -d' ' -f2-
}

OUTCOME="status=failed reason=out_of_attempts"
ABORTS=0
PREV_DONE=-1
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  CKPT=$(latest_ckpt)
  if [ -n "$CKPT" ]; then
    DONE=$(iteration_of "$CKPT")
  elif [ -n "$SEED_CKPT" ]; then
    CKPT="$SEED_CKPT"; DONE=$(iteration_of "$SEED_CKPT")
  else
    CKPT=""; DONE=0
  fi
  if [ $((DONE + 1)) -ge "$TARGET" ]; then
    say "reached $DONE/$TARGET -- done"
    OUTCOME="status=done reached=$DONE target=$TARGET attempts=$((attempt - 1))"
    break
  fi
  # Three consecutive attempts that die without saving anything new are a
  # recipe problem (or a broken machine), not a hiccup.
  if [ "$DONE" -eq "$PREV_DONE" ]; then
    ABORTS=$((ABORTS + 1))
    if [ "$ABORTS" -ge "$ABORT_LIMIT" ]; then
      say "no progress in $ABORTS attempts from $DONE -- giving up"
      OUTCOME="status=failed reason=no_progress reached=$DONE target=$TARGET attempts=$((attempt - 1))"
      break
    fi
  else
    ABORTS=0
  fi
  PREV_DONE=$DONE
  REMAINING=$((TARGET - DONE))
  say "attempt $attempt: from ${CKPT:-scratch} @ $DONE, $REMAINING iterations to go"
  RESUME=()
  [ -n "$CKPT" ] && RESUME=(--resume "$CKPT")
  "$PY" training/scripts/train.py "$@" --name "$RUN_NAME" --log-dir "$RUN_DIR" \
      --max-iterations "$REMAINING" "${RESUME[@]}" >> "$LOG" 2>&1
  CODE=$?
  say "attempt $attempt exited with code=$CODE"
  LAST=$(latest_ckpt); LAST_IT=0
  [ -n "$LAST" ] && LAST_IT=$(iteration_of "$LAST")
  if [ "$CODE" -eq 0 ] || [ $((LAST_IT + 1)) -ge "$TARGET" ]; then
    say "finished at $LAST_IT/$TARGET"
    OUTCOME="status=done reached=$LAST_IT target=$TARGET attempts=$attempt"
    break
  fi
  sleep 30
done
say "supervisor finished: $OUTCOME"
echo "$OUTCOME" > "$RUN_DIR/SUPERVISOR_DONE"
