#!/bin/bash
# One-command Kaggle entrypoint: setup runtime if needed, then train.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/deploy/v2/common.sh"

WORK="${WORK:-$(v2_default_work_dir)}"

if [ ! -d "$WORK/verl" ] || [ ! -f "$WORK/data/train.parquet" ]; then
    bash "$SCRIPT_DIR/deploy/v2/setup.sh"
fi

echo "============================================"
echo "  AlphaAgentEvo v2 GRPO Training"
echo "============================================"
echo "Repo: $SCRIPT_DIR"
echo "Work: $WORK"
echo

exec bash "$SCRIPT_DIR/deploy/v2/train.sh" "$@"
