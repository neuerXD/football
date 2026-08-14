#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$ROOT/.deps/sysroot/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBGL_DRIVERS_PATH="$ROOT/.deps/sysroot/usr/lib/x86_64-linux-gnu/dri"
export __EGL_VENDOR_LIBRARY_DIRS="$ROOT/.deps/sysroot/usr/share/glvnd/egl_vendor.d"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-swrast}"

exec "$ROOT/.venv/bin/python" "$ROOT/gfootball_3d_web.py" "$@"
