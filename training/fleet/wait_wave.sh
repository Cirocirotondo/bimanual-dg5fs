#!/usr/bin/env bash
# Waits for a wave's runs to finish, then pulls them. No model involved: run it
# in the background and let its exit be the one wake-up (CLAUDE.md, rule 2).
#
#   training/fleet/wait_wave.sh tars:m1_fixed_resume tars:m1_adaptive case:m1_fixed ...
#
# A run is finished when its run directory holds SUPERVISOR_DONE. Checks every
# 5 minutes (one ssh per host); an unreachable host is retried, not fatal.
# Prints one summary line per run at the end.
set -u
cd "$(dirname "$0")/../.."
declare -A done_runs=()
while :; do
  pending=0
  for spec in "$@"; do
    host="${spec%%:*}"; run="${spec#*:}"
    [ -n "${done_runs[$spec]:-}" ] && continue
    outcome=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "scirelli@${host}.inf.ethz.ch" \
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
