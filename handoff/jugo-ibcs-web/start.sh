#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "No .venv found. Run ./setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
  echo "Warning: ROBOFLOW_API_KEY is not set. Copy .env.example to .env and add your key."
fi

echo "Starting Jugo IBCS Analysis at http://127.0.0.1:8000"
exec uvicorn web.server:app --host 127.0.0.1 --port 8000
