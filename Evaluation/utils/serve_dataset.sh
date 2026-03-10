#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"
DATA_ROOT="${2:-/opt/dataset}"

python3 -m http.server "${PORT}" --directory "${DATA_ROOT}"
