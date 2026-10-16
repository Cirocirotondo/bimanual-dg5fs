#!/usr/bin/env bash
# Appends a run to a GPU worker's queue (under the queue's lock; see gpu_worker.sh).
#
#   training/fleet/enqueue.sh tars0 l1_fixed 9000 - --num-envs 4096 --reference lift_cube_2 ...
#   training/fleet/enqueue.sh --show            # every queue, the workers and the last events
set -eu
cd "$(dirname "$0")/../.."
dir=training/logs/fleet
mkdir -p "$dir"
if [ "${1:-}" = "--show" ]; then
  for q in "$dir"/*.queue; do
    [ -f "$q" ] || continue
    slot=$(basename "$q" .queue)
    pid=$(cat "$dir/$slot.pid" 2>/dev/null || true)
    alive=down; [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive="up ($pid)"
    echo "== $slot worker $alive"; grep -v '^\s*\(#\|$\)' "$q" | sed 's/^/   /' || true
  done
  echo "== last events"; tail -n 12 "$dir/events.log" 2>/dev/null || true
  exit 0
fi
slot="${1:?slot}"; shift
[ $# -ge 3 ] || { sed -n '2,6p' "$0" >&2; exit 2; }
( flock 9; echo "$*" >> "$dir/$slot.queue" ) 9> "$dir/$slot.queue.lock"
echo "queued on $slot: $*"
