FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    DISPLAY=:99

# Virtual display + VNC bridge so the operator can drive the (headed) Chromium
# that the automation controls, plus the runtime libs OpenCV needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11vnc novnc websockify fluxbox \
        libgl1 libglib2.0-0 \
        ca-certificates curl procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && playwright install --with-deps chromium

COPY . .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 6080
CMD ["/entrypoint.sh"]
