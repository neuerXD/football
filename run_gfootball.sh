#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$ROOT/.deps/sysroot/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBGL_DRIVERS_PATH="$ROOT/.deps/sysroot/usr/lib/x86_64-linux-gnu/dri"
export __EGL_VENDOR_LIBRARY_DIRS="$ROOT/.deps/sysroot/usr/share/glvnd/egl_vendor.d"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-swrast}"
if [ -n "${DISPLAY:-}" ]; then
  export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"
fi

if [ "$#" -eq 0 ]; then
  set -- --players= --level=11_vs_11_official_ai --action_set=full --game_engine_random_seed=8
fi

exec "$ROOT/.venv/bin/python" -m gfootball.play_game "$@"
