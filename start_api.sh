#!/bin/bash
# Start the Kaggle-friendly Verl backtest API on port 8002.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/deploy/v2/common.sh"

WORK="${WORK:-$(v2_default_work_dir)}"

if [ ! -d "$WORK/verl" ]; then
    bash "$SCRIPT_DIR/deploy/v2/setup.sh"
fi

v2_activate_env "$WORK"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "  AlphaAgentEvo v2 Backtest API"
echo "============================================"
echo "Repo: $SCRIPT_DIR"
echo "Work: $WORK"
echo "URL:  http://localhost:8002/health"
echo

exec python -u "$SCRIPT_DIR/deploy/api_server_verl.py"
