#!/usr/bin/env bash
# Waits until the GPU workers log a run more than already seen, then prints the
# new "done" events and exits: the strategist's one wake-up per finished run.
#
#   training/fleet/wait_event.sh            # wait for the next finished run
set -u
cd "$(dirname "$0")/../.."
log=training/logs/fleet/events.log
touch "$log"
seen=$(grep -c ' done ' "$log")
until [ "$(grep -c ' done ' "$log")" -gt "$seen" ]; do sleep 60; done
sleep 5    # a second worker finishing at the same minute comes along
grep ' done ' "$log" | tail -n +"$((seen + 1))"
