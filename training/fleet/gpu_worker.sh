#!/usr/bin/env bash
# Keeps one GPU busy: runs its queue one training at a time, and after each one
# pulls it, evaluates it with videos and rebuilds its wave table, then starts the
# next line at once. No model involved; start one per GPU and leave them running.
#
#   setsid nohup training/fleet/gpu_worker.sh tars0 > /dev/null 2>&1 &
#   slots: tars0 tars1 case0 case1 desktop
#
# Queue: training/logs/fleet/<slot>.queue, one run per line (append with enqueue.sh):
#   NAME TARGET SEED|- <train.py args ...>
# SEED is a checkpoint path relative to bimanual-dg5fs/ on the machine that trains
# (stage it on a server with fleet.sh stage first), or - to start from scratch.
# The wave of a run is the part of NAME before the first "_" (l1_fixed -> L1).
#
# Events (one line each, for whoever is watching): training/logs/fleet/events.log
#   <time> <slot> start|done|failed NAME [outcome, evaluation summary]
# Stop the worker after its current run: touch training/logs/fleet/<slot>.stop
set -u
cd "$(dirname "$0")/../.."
slot="${1:?slot: tars0 tars1 case0 case1 desktop}"
dir=training/logs/fleet
queue="$dir/$slot.queue"
mkdir -p "$dir" training/logs/queue training/logs/waves
touch "$queue"
PY=deps/IsaacLab/.venv/bin/python
case "$slot" in
  tars0|tars1|case0|case1) host="${slot%[01]}"; gpu="${slot: -1}"; HOSTNAME_FULL="$host.inf.ethz.ch" ;;
  desktop) host=desktop ;;
  *) echo "unknown slot $slot" >&2; exit 2 ;;
esac

event() { echo "$(TZ=Europe/Zurich date '+%Y-%m-%d %H:%M:%S') $slot $*" >> "$dir/events.log"; }

pop() {   # first queue line, removed under the queue's lock
  (
    flock 9
    line=$(grep -v '^\s*\(#\|$\)' "$queue" | head -1)
    if [ -n "$line" ]; then
      grep -vxF -- "$line" "$queue" > "$queue.tmp"; mv "$queue.tmp" "$queue"
    fi
    printf '%s' "$line"
  ) 9> "$queue.lock"
}

remote() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOSTNAME_FULL" "$@" 2>/dev/null; }

echo "$$" > "$dir/$slot.pid"
event "worker up (pid $$)"
while :; do
  [ -f "$dir/$slot.stop" ] && { rm -f "$dir/$slot.stop"; event "worker stopped"; exit 0; }
  line=$(pop)
  if [ -z "$line" ]; then sleep 60; continue; fi
  read -r name target seed rest <<< "$line"
  args="${rest#-- }"
  seed_args=()
  [ "$seed" != "-" ] && seed_args=(--seed-checkpoint "$seed")
  event "start $name target=$target seed=$seed args: $args"
  if [ "$host" = desktop ]; then
    # shellcheck disable=SC2086
    training/fleet/supervise_train.sh --python "$PY" --run-name "$name" --target "$target" "${seed_args[@]}" -- $args
    run_dir=$(cat "training/logs/queue/$name.rundir" 2>/dev/null)
    outcome=$(cat "$run_dir/SUPERVISOR_DONE" 2>/dev/null || echo "status=killed")
  else
    # shellcheck disable=SC2086
    if ! HOST="$HOSTNAME_FULL" training/fleet/fleet.sh "$gpu" supervise --run-name "$name" --target "$target" \
         "${seed_args[@]}" -- $args > "$dir/$slot.launch.log" 2>&1; then
      # Most often the GPU is still busy (someone else's job, or a run not launched by
      # this worker): put the run back at the head of the queue and try again later.
      ( flock 9; { echo "$line"; cat "$queue"; } > "$queue.tmp"; mv "$queue.tmp" "$queue" ) 9> "$queue.lock"
      event "waiting $name: $(grep -v '^\s*$' "$dir/$slot.launch.log" | grep -v 'inf.ethz.ch - \|^---' | head -1)"
      sleep 300; continue
    fi
    outcome=""
    while [ -z "$outcome" ]; do
      sleep 180
      outcome=$(remote "d=\$(sed 's|^/workspace/bimanual-dg5fs|bimanual-dg5fs|' bimanual-dg5fs/training/logs/queue/$name.rundir 2>/dev/null); \
                        cat \"\$d/SUPERVISOR_DONE\" 2>/dev/null" | grep '^status=' || true)
    done
    HOST="$HOSTNAME_FULL" training/fleet/fleet.sh pull "*_$name" > /dev/null 2>&1 || event "pull failed $name"
    up=$(tr '[:lower:]' '[:upper:]' <<< "$host")
    run_dir=$(ls -d training/logs/motion_imitation/${up}_*_"$name" 2>/dev/null | tail -1)
  fi
  # Evaluate in the background (one evaluation at a time on the desktop GPU) so the
  # next run starts at once instead of waiting for the evaluation queue.
  (
    summary=""
    if [ -n "$run_dir" ] && ls "$run_dir"/model_*.pt > /dev/null 2>&1; then
      flock "$dir/eval.lock" env -u PYTHONPATH "$PY" training/scripts/evaluate_run.py "$run_dir" \
        > "$run_dir/evaluate.log" 2>&1 || event "evaluation failed $name ($run_dir/evaluate.log)"
      summary=$(grep -a '^-- ' "$run_dir/evaluate.log" | head -1 | sed 's/^-- //')
      wave=$(tr '[:lower:]' '[:upper:]' <<< "${name%%_*}")
      wave_runs=$(ls -d training/logs/motion_imitation/*_"${name%%_*}"_* 2>/dev/null | while read -r d; do
                    ls "$d"/eval/*/summary.md > /dev/null 2>&1 && echo "$d"; done)
      # shellcheck disable=SC2086
      [ -n "$wave_runs" ] && flock "$dir/eval.lock" training/fleet/evaluate_wave.sh "$wave" $wave_runs \
        > "training/logs/waves/${wave}_evaluate.log" 2>&1
    fi
    event "done $name $outcome dir=$run_dir | $summary"
  ) &
done
