#!/usr/bin/env bash
# Waits for a wave's runs to finish, then pulls them. No model involved: run it
# in the background and let its exit be the one wake-up (CLAUDE.md, rule 2).
#
#   training/fleet/wait_wave.sh [--wave M1] tars:m1_fixed_resume tars:m1_adaptive case:m1_fixed ...
#
# With --wave, the pulled runs are then evaluated on the desktop GPU with videos
# (evaluate_wave.sh) and the wave table goes to training/logs/waves/<wave>.md.
#
# A run is finished when its run directory holds SUPERVISOR_DONE. Checks every
# 5 minutes (one ssh per host); an unreachable host is retried, not fatal.
# Prints one summary line per run at the end.
set -u
cd "$(dirname "$0")/../.."
wave=""
if [ "${1:-}" = "--wave" ]; then wave="$2"; shift 2; fi
declare -A done_runs=()
while :; do
  pending=0
  for spec in "$@"; do
    host="${spec%%:*}"; run="${spec#*:}"
    [ -n "${done_runs[$spec]:-}" ] && continue
    outcome=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${host}.inf.ethz.ch" \
      "d=\$(cat bimanual-dg5fs/training/logs/queue/${run}.rundir 2>/dev/null); \
       d=\${d/\/workspace\/bimanual-dg5fs/bimanual-dg5fs}; cat \"\$d/SUPERVISOR_DONE\" 2>/dev/null" 2>/dev/null \
      | grep '^status=' || true)
    if [ -n "${outcome}" ]; then done_runs[$spec]="${outcome}"; else pending=$((pending + 1)); fi
  done
  [ "${pending}" -eq 0 ] && break
  sleep 300
done
for spec in "$@"; do
  host="${spec%%:*}"; run="${spec#*:}"
  HOST="${host}.inf.ethz.ch" training/fleet/fleet.sh pull "*_${run}" > /dev/null 2>&1 || echo "pull failed: ${spec}"
  echo "${spec}: ${done_runs[$spec]}"
done
if [ -n "${wave}" ]; then
  runs=()
  for spec in "$@"; do
    host="${spec%%:*}"; run="${spec#*:}"
    up="$(tr '[:lower:]' '[:upper:]' <<< "${host}")"
    for d in training/logs/motion_imitation/${up}_*_"${run}"; do [ -d "$d" ] && runs+=("$d"); done
  done
  [ ${#runs[@]} -eq 0 ] || training/fleet/evaluate_wave.sh "${wave}" "${runs[@]}" > "training/logs/waves/${wave}_evaluate.log" 2>&1 \
    && echo "wave table: training/logs/waves/${wave}.md" || echo "evaluation failed: training/logs/waves/${wave}_evaluate.log"
fi
