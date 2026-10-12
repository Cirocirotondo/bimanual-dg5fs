#!/usr/bin/env bash
# Starts Nando's MANUS SDK client (ZMQ build): Glove ergonomics on tcp 8001,
# skeletons on tcp 8000. Choose [1] Integrated when asked.
#
# The client's libManusSDK.so needs gRPC 1.28.1 / protobuf 3.11 (libgrpc.so.9,
# libgrpc++.so.1, libprotobuf.so.22), which Ubuntu 22.04 does not ship; copies
# built on the UR5 PC live in ./lib (git-ignored).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CLIENT_DIR="${MANUS_CLIENT_DIR:-/home/simone/projects/robohand/src/manus/SDKClient_Linux}"
cd "$CLIENT_DIR"   # the client finds libManusSDK.so through the relative runpath ./ManusSDK/lib
export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec ./SDKClient_Linux.out "$@"
