#!/usr/bin/env bash
# Foreground runner for systemd. Sources an env file then exec uvicorn
# so systemd owns the process directly (matches Type=simple).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <env-file>" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$1"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "env file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

PORT="${PORT:-8000}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"

"$ROOT_DIR/scripts/bootstrap_venv.sh"

if [[ -n "${HF_CACHE_DIR:-}" ]]; then
  mkdir -p "$ROOT_DIR/${HF_CACHE_DIR#./}"
  export HF_HOME="$ROOT_DIR/${HF_CACHE_DIR#./}"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "$ROOT_DIR/.venv/bin/uvicorn" provider.app:app \
  --app-dir "$ROOT_DIR" \
  --host "$BIND_HOST" \
  --port "$PORT"
