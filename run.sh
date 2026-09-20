#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for cand in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && \
     "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer was not found. Install it, then run this again." >&2
  exit 1
fi
echo "  Using: $($PY -c 'import sys;print(sys.executable)')"
"$PY" -m pip install --disable-pip-version-check --quiet -r requirements.txt
exec "$PY" server.py
