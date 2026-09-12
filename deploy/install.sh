#!/usr/bin/env bash
#
# Install the inference server as a systemd service.
#
# The deployed artifact is a PyInstaller binary, not a container image. This
# script lays out /opt/mlops, installs the unit, and verifies the service
# actually serves before declaring success.
#
# Usage:
#   sudo deploy/install.sh [--prefix /opt/mlops] [--port 8000] [--no-start]
#
# Expects to be run from a checkout that has already been built:
#   make build-binaries && sudo deploy/install.sh
set -euo pipefail

PREFIX="/opt/mlops"
SERVICE_USER="mlops"
PORT="8000"
START=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)   PREFIX="$2"; shift 2 ;;
    --port)     PORT="$2"; shift 2 ;;
    --no-start) START=0; shift ;;
    -h|--help)  sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$ROOT/dist/server_module"
UNIT_SRC="$ROOT/deploy/mlops-server.service"
UNIT_DST="/etc/systemd/system/mlops-server.service"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "must run as root (try: sudo $0)" >&2
  exit 1
fi

if [[ ! -x "$BINARY" ]]; then
  echo "server binary not found or not executable: $BINARY" >&2
  echo "build it first:  make build-binaries" >&2
  exit 1
fi

echo "==> verifying the binary before installing it"
bash "$ROOT/scripts/smoke_test.sh" "$BINARY"

echo "==> creating service account '$SERVICE_USER'"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

echo "==> laying out $PREFIX"
install -d -o root -g root -m 755 "$PREFIX" "$PREFIX/bin" "$PREFIX/config"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 750 "$PREFIX/data"

install -o root -g root -m 755 "$BINARY" "$PREFIX/bin/server_module"
install -o root -g root -m 644 "$ROOT/config/runtime.json" "$PREFIX/config/runtime.json"
install -o root -g root -m 644 "$ROOT/README.md" "$PREFIX/README.md"

# Never overwrite an existing .env: it holds this host's paths and secrets.
if [[ -f "$PREFIX/.env" ]]; then
  echo "    keeping existing $PREFIX/.env"
else
  echo "    writing $PREFIX/.env from deploy/env.example"
  sed -e "s|^PORT=.*|PORT=$PORT|" \
      -e "s|^REPOSITORY_DATA_PATH=.*|REPOSITORY_DATA_PATH=$PREFIX/data/example.db|" \
      -e "s|^REPOSITORY_OBJECT_PATH=.*|REPOSITORY_OBJECT_PATH=$PREFIX/data/example.db|" \
      "$ROOT/deploy/env.example" > "$PREFIX/.env"
  chown root:root "$PREFIX/.env"
  chmod 600 "$PREFIX/.env"
fi

# The registry is the whole system: every trained model and every
# preprocessing pipeline lives in this one file.
DB="$PREFIX/data/example.db"
if [[ -f "$DB" ]]; then
  echo "    registry already present: $DB"
else
  echo
  echo "    NOTE: no registry at $DB yet."
  echo "    The service will refuse to start until one exists, by design --"
  echo "    a server that loads zero models is not useful. Copy one in:"
  echo "        install -o $SERVICE_USER -g $SERVICE_USER -m 640 <your>.db $DB"
  echo
fi

echo "==> installing the systemd unit"
install -o root -g root -m 644 "$UNIT_SRC" "$UNIT_DST"
systemctl daemon-reload
systemctl enable mlops-server.service >/dev/null

if [[ "$START" -eq 0 ]]; then
  echo "==> --no-start given; not starting the service"
  exit 0
fi

if [[ ! -f "$DB" ]]; then
  echo "==> not starting: no registry to serve"
  exit 0
fi

echo "==> starting mlops-server"
systemctl restart mlops-server.service

echo "==> waiting for /health"
for _ in $(seq 1 30); do
  if body="$(curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null)"; then
    echo "    $body"
    if printf '%s' "$body" | grep -q '"status":"ok"'; then
      count="$(printf '%s' "$body" | sed -n 's/.*"model_count":\([0-9]*\).*/\1/p')"
      echo
      echo "mlops-server is up on port $PORT with ${count:-?} model(s)."
      exit 0
    fi
    echo "    service reports a non-ok status; check: journalctl -u mlops-server -n 50" >&2
    exit 1
  fi
  sleep 1
done

echo "service did not become healthy; check: journalctl -u mlops-server -n 50" >&2
exit 1
