#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    major="${version%%.*}"
    minor="${version#*.}"
    if [[ "$major" -eq 3 && "$minor" -ge 10 && "$minor" -le 12 ]]; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "Python 3.10–3.12 is required. Install Python 3.12 and try again."
  exit 1
fi

echo "Using $PYTHON ($("$PYTHON" --version))"

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo ""
  echo "Created .env from .env.example — add your ROBOFLOW_API_KEY before analyzing images."
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Edit .env and set ROBOFLOW_API_KEY"
echo "  2. Run: ./start.sh"
echo "  3. Open: http://127.0.0.1:8000"
