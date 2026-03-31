#!/bin/bash
# AlphaAgentEvo v2 — post-train inference smoke test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

REPO_ROOT="$(v2_repo_root)"
WORK="${WORK:-$(v2_default_work_dir)}"
LOGS="$WORK/logs"
DATA="$WORK/data"
ENV_NAME="${ENV_NAME:-verl041}"

API_BASE_URL="${API_BASE_URL:-http://localhost:8002}"
HEALTH_URL="${API_BASE_URL%/}/health"
export ALPHAEVO_BACKTEST_API_URL="${ALPHAEVO_BACKTEST_API_URL:-$API_BASE_URL/backtest}"

INFER_MAX_SEEDS="${INFER_MAX_SEEDS:-3}"
INFER_MAX_TURNS="${INFER_MAX_TURNS:-1}"
INFER_MAX_NEW_TOKENS="${INFER_MAX_NEW_TOKENS:-512}"
INFER_MAX_TOOL_CALLS_PER_TURN="${INFER_MAX_TOOL_CALLS_PER_TURN:-4}"
INFER_PRINT_IO="${INFER_PRINT_IO:-1}"
INFER_MAX_LOG_CHARS="${INFER_MAX_LOG_CHARS:-8000}"
INFER_LABEL="${INFER_LABEL:-posttrain-smoke}"

v2_activate_env "$WORK"
mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORK:$WORK/verl:$WORK/backtest:$WORK/expression_manager:$REPO_ROOT:${PYTHONPATH:-}"

LATEST_HF="$(ls -dt "$WORK"/verl/checkpoints/alphaagentevo-v2/*/global_step_*/actor/huggingface 2>/dev/null | head -n1 || true)"
if [ -z "$LATEST_HF" ] || [ ! -d "$LATEST_HF" ]; then
    echo "ERROR: No checkpoint huggingface dir found under $WORK/verl/checkpoints/alphaagentevo-v2" >&2
    exit 1
fi

if ! curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    if [ "${AUTO_START_API:-1}" = "1" ]; then
        nohup python -u "$REPO_ROOT/deploy/api_server_verl.py" > "$LOGS/api.log" 2>&1 &
        for _ in $(seq 1 60); do
            sleep 2
            if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
                break
            fi
        done
    fi
fi
curl -sf "$HEALTH_URL" >/dev/null 2>&1 || {
    echo "ERROR: Backtest API not healthy at $HEALTH_URL" >&2
    exit 1
}

VAL_PARQUET="$DATA/val.parquet"
SMOKE_PARQUET="$DATA/val_smoke_${INFER_MAX_SEEDS}.parquet"
python - <<PY
import pandas as pd
src = "$VAL_PARQUET"
dst = "$SMOKE_PARQUET"
n = int("$INFER_MAX_SEEDS")
df = pd.read_parquet(src).head(n).copy()
df.to_parquet(dst, index=False)
print(f"Prepared {len(df)} rows: {dst}")
PY

echo "============================================================"
echo "Starting post-train inference smoke"
echo "  Model: $LATEST_HF"
echo "  Data:  $SMOKE_PARQUET"
echo "  Turns: $INFER_MAX_TURNS"
echo "  Max tool calls/turn: $INFER_MAX_TOOL_CALLS_PER_TURN"
echo "  Print I/O: $INFER_PRINT_IO"
echo "============================================================"

PRINT_IO_ARGS=()
if [ "$INFER_PRINT_IO" = "1" ]; then
    PRINT_IO_ARGS+=(--print-io)
fi

python -u "$REPO_ROOT/training/evaluate.py" \
  --base-model "$LATEST_HF" \
  --data "$SMOKE_PARQUET" \
  --max-turns "$INFER_MAX_TURNS" \
  --max-new-tokens "$INFER_MAX_NEW_TOKENS" \
  --max-tool-calls-per-turn "$INFER_MAX_TOOL_CALLS_PER_TURN" \
  --max-log-chars "$INFER_MAX_LOG_CHARS" \
  "${PRINT_IO_ARGS[@]}" \
  --label "$INFER_LABEL" | tee "$LOGS/infer.log"

echo "Inference log: $LOGS/infer.log"
