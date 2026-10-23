#!/usr/bin/env bash
set -euo pipefail

mkdir -p input output logs vault profiles

# --- stale Chromium profile locks -----------------------------------------
# The Chrome profile lives on a volume, so a container that is restarted (or
# killed) leaves behind singleton locks naming a PID and hostname from the
# previous container. Chromium then refuses to launch and the worker dies at
# boot. Nothing inside a freshly started container can be holding them, so
# clearing them here is safe and is the only way a restart comes up clean.
find profiles -maxdepth 2 \( -name 'Singleton*' \) -delete 2>/dev/null || true

# --- virtual display -------------------------------------------------------
rm -f /tmp/.X99-lock
Xvfb :99 -screen 0 "${SCREEN_SIZE:-1600x1000x24}" -nolisten tcp &
for _ in $(seq 1 40); do
    [ -e /tmp/.X11-unix/X99 ] && break
    sleep 0.25
done
fluxbox >/dev/null 2>&1 &

# --- VNC + noVNC (operator signs in to ChatGPT through this) ---------------
if [ -n "${VNC_PASSWORD:-}" ]; then
    x11vnc -storepasswd "$VNC_PASSWORD" /tmp/vncpass >/dev/null 2>&1
    x11vnc -display :99 -rfbport 5900 -rfbauth /tmp/vncpass -forever -shared -quiet -bg
else
    echo "WARNING: VNC_PASSWORD not set - remote desktop is unauthenticated." >&2
    x11vnc -display :99 -rfbport 5900 -nopw -forever -shared -quiet -bg
fi
websockify --web=/usr/share/novnc 6080 localhost:5900 >/dev/null 2>&1 &

# --- database: wait, then migrate -----------------------------------------
# The db service has a healthcheck and the app `depends_on` it being healthy,
# but we still wait+retry here so the app never starts against a db that is
# mid-init, and so local-dev (no compose healthcheck) also comes up cleanly.
# `alembic upgrade head` creates/updates every table automatically on each
# deploy — no manual SQL step. A failed migration aborts startup loudly rather
# than running the app against a half-migrated schema.
echo "[entrypoint] waiting for database…"
python - <<'PY'
import sys
from db.session import wait_for_db
sys.exit(0 if wait_for_db(max_seconds=90) else 1)
PY
if [ $? -ne 0 ]; then
    echo "[entrypoint] database never became reachable; aborting." >&2
    exit 1
fi

echo "[entrypoint] running alembic upgrade head…"
alembic upgrade head

# --- one-time, idempotent imports (users/tokens, vault folders, prompts) ---
# Safe to run on every boot: each importer is a no-op once its data is present.
echo "[entrypoint] running first-start imports (idempotent)…"
python -m scripts.first_start_import || echo "[entrypoint] import step reported an issue (continuing)"

# --- app -------------------------------------------------------------------
exec uvicorn app:app --host 0.0.0.0 --port 8000
