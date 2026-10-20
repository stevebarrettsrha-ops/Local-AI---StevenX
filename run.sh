#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Started from the file manager (right-click → Run as a Program) there is no
# terminal: nothing this script says can be seen, and a failure just looks like
# nothing happening. On a Linux desktop, open a terminal and run in it instead.
if [ "$(uname)" = "Linux" ] && [ -z "${LLAMA_STUDIO_IN_TERMINAL:-}" ] \
   && [ ! -t 0 ] && [ ! -t 1 ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
  export LLAMA_STUDIO_IN_TERMINAL=1
  SELF="$(pwd)/$(basename "$0")"
  for term in gnome-terminal ptyxis kgx konsole xfce4-terminal mate-terminal \
              tilix x-terminal-emulator xterm; do
    command -v "$term" >/dev/null 2>&1 || continue
    case "$term" in
      gnome-terminal|ptyxis|kgx) "$term" -- bash "$SELF" && exit 0 ;;
      xfce4-terminal|mate-terminal) "$term" -x bash "$SELF" && exit 0 ;;
      *) "$term" -e bash "$SELF" && exit 0 ;;
    esac
  done
  # No terminal found: carry on without one. The page still opens.
fi

# In a window we opened, keep it open on failure so the reason can be read.
# (That is why the server is not started with `exec` below: the trap must
# outlive it.)
pause_on_error() {
  local code=$?
  if [ -n "${LLAMA_STUDIO_IN_TERMINAL:-}" ] && [ "$code" -ne 0 ] \
     && [ "$code" -ne 130 ] && [ -t 0 ]; then
    printf "\n  Llama Studio stopped (exit code %s). Press Enter to close." "$code"
    read -r _ || true
  fi
}
trap pause_on_error EXIT

PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && \
     "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer was not found. Install it, then run this again." >&2
  exit 1
fi
# The studio's own venv (Flask, requests, the Office libraries). Current
# Debian/Ubuntu refuse pip into the system Python (PEP 668). The check asks for
# pip, not just a Python: without python3-venv, `venv` dies after writing
# bin/python but before pip, and that half-built folder is rebuilt, not kept.
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
  echo "  Setting up Llama Studio's packages (first run only)..."
  "$PY" -m venv .venv || {
    rm -rf .venv
    echo "Could not create a virtual environment. On Debian/Ubuntu install" >&2
    echo "python3-venv (sudo apt-get install -y python3-venv), then run this again." >&2
    exit 1
  }
fi
PY=.venv/bin/python
echo "  Using: $($PY -c 'import sys;print(sys.executable)')"
"$PY" -m pip install --disable-pip-version-check --quiet -r requirements.txt
"$PY" server.py
