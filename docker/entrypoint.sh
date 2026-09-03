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

# --- app -------------------------------------------------------------------
exec uvicorn app:app --host 0.0.0.0 --port 8000
