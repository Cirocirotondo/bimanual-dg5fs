#!/usr/bin/env bash
# Trainings on the tars / case servers (rootless docker, one container per GPU).
# Port of simtoolreal_newton's tars_train.sh + tars_sync.sh. Run from bimanual-dg5fs/.
#
#   HOST=tars.inf.ethz.ch training/fleet/fleet.sh push                 # code, USDs, References -> ~/bimanual-dg5fs
#   training/fleet/fleet.sh stage <checkpoint.pt>                      # a seed checkpoint -> same path on the server
#   training/fleet/fleet.sh <gpu> supervise --run-name N --target T [--seed-checkpoint P] -- <train.py args>
#   training/fleet/fleet.sh <gpu> exec <command ...>                   # one-off job (evaluation) in a container
#   training/fleet/fleet.sh status | log <container|run-name> [lines] | stop <container>
#   training/fleet/fleet.sh pull [run-dir-name]                        # light files (+ newest checkpoint) of the runs
#
# HOST defaults to tars.inf.ethz.ch; case.inf.ethz.ch is the other server.
#
# The image is simtoolreal_newton's str_newton:v2: built from the same Isaac Lab
# commit and lock (deps/IsaacLab a8b4da3), so a checkpoint trained there loads here.
# The project is not in the image; the rsync mirror is bind-mounted and
# supervise_train.sh puts its packages on PYTHONPATH.
set -eu

host="${HOST:-tars.inf.ethz.ch}"
short="$(cut -d. -f1 <<< "${host}")"
remote_root="bimanual-dg5fs"
workspace="/workspace/bimanual-dg5fs"
image="str_newton:v2"
prefix="bim_gpu"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=15)
repo="$(cd "$(dirname "$0")/../.." && pwd)"

remote() { ssh "${ssh_opts[@]}" "${host}" "source ~/.docker_env.sh; $*"; }

docker_run() {
  local name="$1" gpu="$2"; shift 2
  # One-off exec containers remove themselves; their output stays in the launch log.
  local rm=""
  case "${name}" in *_exec_*) rm="--rm" ;; esac
  local payload
  # Quoted once here, base64 through ssh and docker, decoded by bash in the container.
  payload="$(printf '%q ' "$@"; printf '>> training/logs/launch/%s.log 2>&1' "${name}")"
  payload="$(printf '%s' "${payload}" | base64 -w0)"
  remote "mkdir -p ${remote_root}/training/logs/launch ${remote_root}/.warp_cache && docker rm ${name} >/dev/null 2>&1; \
    docker run -d ${rm} --name ${name} \
      --runtime=nvidia --gpus device=${gpu} --ipc=host --ulimit stack=67108864 --ulimit core=0 --user root \
      -e TZ=Europe/Zurich -e NVIDIA_DISABLE_REQUIRE=1 -e FLEET_HOST=${short} \
      -v ~/${remote_root}:${workspace} \
      -v ~/${remote_root}/.warp_cache:/root/.cache/warp \
      -v /usr/share/zoneinfo:/usr/share/zoneinfo:ro \
      -w ${workspace} \
      ${image} bash -c 'echo ${payload} | base64 -d | bash'"
}

# A GPU is ours only if nobody (our containers, simtoolreal's, other users) uses it.
refuse_busy_gpu() {
  local used
  used="$(remote "nvidia-smi --id=${1} --query-gpu=memory.used --format=csv,noheader,nounits" | tr -d ' ')"
  if [ "${used}" -gt 1000 ]; then
    echo "GPU ${1} on ${host} is in use (${used} MiB):" >&2
    remote "docker ps --format '{{.Names}} {{.Status}}'; nvidia-smi --id=${1} --query-compute-apps=pid,used_memory --format=csv" >&2
    exit 1
  fi
}

case "${1:-}" in
  push)
    cd "${repo}"
    remote "mkdir -p ${remote_root}/training/logs/queue ${remote_root}/training/logs/launch ${remote_root}/.warp_cache"
    rsync -a --delete -e "ssh ${ssh_opts[*]}" \
      --exclude ".git" --exclude "deps" --exclude "teleoperation" --exclude "training/logs" \
      --exclude "__pycache__" --exclude "*.pyc" --exclude ".pytest_cache" --exclude "*.egg-info" \
      --exclude ".warp_cache" --exclude ".venv" \
      ./ "${host}:${remote_root}/"
    echo "pushed ${repo} -> ${host}:${remote_root}"
    ;;
  stage)
    ckpt="${2:?checkpoint path, relative to bimanual-dg5fs/}"
    cd "${repo}"
    rsync -a --mkpath -e "ssh ${ssh_opts[*]}" "${ckpt}" "${host}:${remote_root}/${ckpt}"
    echo "staged ${ckpt} on ${host}"
    ;;
  status)
    remote "hostname; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo; \
      docker ps -a --filter name=${prefix} --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'; echo; \
      for f in ${remote_root}/training/logs/queue/*.log; do [ -f \"\$f\" ] || continue; echo \"== \$(basename \$f)\"; \
        grep '^\[sup\]' \"\$f\" | tail -2; grep -a 'Learning iteration' \"\$f\" | tail -1; done"
    ;;
  log)
    # A run name shows the supervisor + trainer log; a container name its launch log.
    name="${2:?container or run name}"
    remote "f=${remote_root}/training/logs/queue/${name}.log; [ -f \$f ] || f=${remote_root}/training/logs/launch/${name}.log; tail -n ${3:-40} \$f"
    ;;
  stop)
    # Stopping kills the supervisor before it can write SUPERVISOR_DONE; write it
    # here so wait_wave.sh does not wait for that run forever.
    name="${2:?container name}"
    remote "docker stop ${name} && docker rm ${name}"
    run="${name#${prefix}[0-9]_}"
    remote "d=\$(sed 's|^${workspace}|${remote_root}|' ${remote_root}/training/logs/queue/${run}.rundir 2>/dev/null); \
      if [ -n \"\$d\" ] && [ ! -f \"\$d/SUPERVISOR_DONE\" ]; then echo status=stopped > \"\$d/SUPERVISOR_DONE\"; fi"
    ;;
  pull)
    # Light files of every run (or one), plus each run's newest checkpoint, into
    # training/logs/motion_imitation/<HOST>_<run>; the queue logs into training/logs/queue/<host>/.
    cd "${repo}"
    pattern="${2:-*}"
    up="$(tr '[:lower:]' '[:upper:]' <<< "${short}")"
    mkdir -p "training/logs/queue/${short}"
    rsync -a --timeout=180 -e "ssh ${ssh_opts[*]}" "${host}:${remote_root}/training/logs/queue/" "training/logs/queue/${short}/" || true
    for run in $(remote "cd ${remote_root}/training/logs/motion_imitation 2>/dev/null && ls -d ${pattern} 2>/dev/null" || true); do
      newest="$(remote "ls ${remote_root}/training/logs/motion_imitation/${run}/model_*.pt 2>/dev/null | sed -n 's/.*model_\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1")"
      dest="training/logs/motion_imitation/${up}_${run}"
      mkdir -p "${dest}"
      rsync -a --timeout=180 --prune-empty-dirs -e "ssh ${ssh_opts[*]}" \
        --include "config.json" --include "SUPERVISOR_DONE" --include "events.out.tfevents.*" \
        ${newest:+--include "model_${newest}.pt"} --exclude "*" \
        "${host}:${remote_root}/training/logs/motion_imitation/${run}/" "${dest}/"
      echo "pulled ${run} -> ${dest} (newest checkpoint ${newest:-none})"
    done
    ;;
  0|1)
    gpu="$1"; shift
    [ $# -ge 1 ] || { echo "no arguments given" >&2; exit 1; }
    if [ "$1" = "exec" ]; then
      shift
      name="${prefix}${gpu}_exec_$(date +%Y%m%d_%H%M%S)_$$"
      [ "${SHARE_GPU:-0}" = "1" ] || refuse_busy_gpu "${gpu}"
      docker_run "${name}" "${gpu}" env PYTHONPATH="${workspace}/src:${workspace}/training" "$@"
      echo "launched ${name} on ${host} GPU ${gpu}: $*"
      echo "  follow: HOST=${host} training/fleet/fleet.sh log ${name}"
      exit 0
    fi
    [ "$1" = "supervise" ] || { echo "use: <gpu> supervise ... or <gpu> exec ..." >&2; exit 1; }
    shift
    run_name="$(sed -n 's/.*--run-name[= ]\([^ ]*\).*/\1/p' <<< "$*")"
    [ -n "${run_name}" ] || { echo "the arguments must carry --run-name" >&2; exit 1; }
    name="${prefix}${gpu}_${run_name}"
    refuse_busy_gpu "${gpu}"
    docker_run "${name}" "${gpu}" bash training/fleet/supervise_train.sh "$@"
    echo "launched ${name} on ${host} GPU ${gpu}"
    echo "  status: HOST=${host} training/fleet/fleet.sh status"
    ;;
  *)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
