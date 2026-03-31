#!/bin/bash
# One-command Kaggle cell launcher for AAE v2 training on H100.
set -euo pipefail

find_repo() {
    local c
    for c in \
        /kaggle/input/datasets/gplebih/aae-new \
        /kaggle/input/datasets/giaphlm/aae-new \
        /kaggle/input/aae-new \
        /kaggle/input/*/aae-new; do
        if [ -d "$c/deploy/v2" ]; then
            printf '%s\n' "$c"
            return 0
        fi
    done
    return 1
}

kill_pat() {
    local pat="$1"
    local target_pid
    local cur
    local parent

    is_ancestor_pid() {
        local candidate="$1"
        cur="$$"
        while [ -n "$cur" ] && [ "$cur" -gt 1 ] 2>/dev/null; do
            [ "$candidate" = "$cur" ] && return 0
            parent="$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d '[:space:]')"
            [ -n "$parent" ] || break
            cur="$parent"
        done
        return 1
    }

    mapfile -t pids < <(pgrep -f "$pat" || true)
    for target_pid in "${pids[@]}"; do
        is_ancestor_pid "$target_pid" && continue
        kill -TERM "$target_pid" 2>/dev/null || true
    done
}

REPO="${REPO:-$(find_repo || true)}"
WORK="${WORK:-/kaggle/working/aae_v2}"

[ -n "$REPO" ] || { echo "ERROR: cannot find dataset repo path (aae-new) under /kaggle/input"; exit 1; }
[ -d "$REPO/deploy/v2" ] || { echo "ERROR: invalid REPO=$REPO"; exit 1; }

mkdir -p /kaggle/working
cat > /kaggle/working/aae_env.sh <<EOF
export REPO="$REPO"
export WORK="$WORK"
EOF

echo "REPO=$REPO"
echo "WORK=$WORK"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

if [ "${RUN_CLEANUP:-1}" = "1" ]; then
    kill_pat "deploy/v2/train.sh"
    kill_pat "verl.trainer.main_ppo"
    kill_pat "sglang::"
    kill_pat "api_server_verl.py"
    sleep 2
    ray stop --force || true
fi

if [ "${RUN_SETUP:-1}" = "1" ]; then
    rm -rf "$WORK"
    mkdir -p "$WORK/logs"
    PYTHONUNBUFFERED=1 bash "$REPO/deploy/v2/setup.sh" 2>&1 | tee "$WORK/logs/setup.live.log"
fi

TOTAL_STEPS="${TOTAL_STEPS:-50}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B-MLX-bf16}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen3-0.6B-MLX-bf16-verl-h100-paperish-50step}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-3}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-2}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-3}"
MAX_TOOL_CALLS_PER_TURN="${MAX_TOOL_CALLS_PER_TURN:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-8192}"
ROLLOUT_MAX_SEQS="${ROLLOUT_MAX_SEQS:-8}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-2048}"
FORWARD_MAX_TOKEN_LEN_PER_GPU="${FORWARD_MAX_TOKEN_LEN_PER_GPU:-2048}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-2048}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
CRITIC_FORWARD_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_FORWARD_MICRO_BATCH_SIZE_PER_GPU:-4}"
CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-5}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
RESUME_MODE="${RESUME_MODE:-disable}"
ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE="${ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE:-False}"
TRAIN_HEARTBEAT_SEC="${TRAIN_HEARTBEAT_SEC:-10}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
ROLLOUT_DISABLE_LOG_STATS="${ROLLOUT_DISABLE_LOG_STATS:-False}"
TRAIN_STDOUT_MODE="${TRAIN_STDOUT_MODE:-pretty}"
TRAIN_STDOUT_FILTER="${TRAIN_STDOUT_FILTER:-Starting AlphaAgentEvo|heartbeat|step:|critic/rewards/mean|val-core/alphaagentevo/reward/mean@3|val-paper/|\\[FactorTool\\] backtest success|\\[factor-live\\]|\\[response-live\\]|TRAIN SUMMARY|ERROR|RayTaskError}"

PYTHONUNBUFFERED=1 \
MODEL="$MODEL" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
TOTAL_STEPS="$TOTAL_STEPS" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
ROLLOUT_N="$ROLLOUT_N" \
VAL_ROLLOUT_N="$VAL_ROLLOUT_N" \
MAX_ASSISTANT_TURNS="$MAX_ASSISTANT_TURNS" \
MAX_TOOL_CALLS_PER_TURN="$MAX_TOOL_CALLS_PER_TURN" \
MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
ROLLOUT_MAX_BATCHED_TOKENS="$ROLLOUT_MAX_BATCHED_TOKENS" \
ROLLOUT_MAX_SEQS="$ROLLOUT_MAX_SEQS" \
PPO_MAX_TOKEN_LEN_PER_GPU="$PPO_MAX_TOKEN_LEN_PER_GPU" \
FORWARD_MAX_TOKEN_LEN_PER_GPU="$FORWARD_MAX_TOKEN_LEN_PER_GPU" \
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU" \
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
CRITIC_FORWARD_MICRO_BATCH_SIZE_PER_GPU="$CRITIC_FORWARD_MICRO_BATCH_SIZE_PER_GPU" \
CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU="$CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU" \
GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
SAVE_FREQ="$SAVE_FREQ" \
TEST_FREQ="$TEST_FREQ" \
VAL_BEFORE_TRAIN="$VAL_BEFORE_TRAIN" \
RESUME_MODE="$RESUME_MODE" \
ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE="$ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE" \
TRAIN_HEARTBEAT_SEC="$TRAIN_HEARTBEAT_SEC" \
ROLLOUT_ENABLE_CHUNKED_PREFILL="$ROLLOUT_ENABLE_CHUNKED_PREFILL" \
ROLLOUT_DISABLE_LOG_STATS="$ROLLOUT_DISABLE_LOG_STATS" \
TRAIN_STDOUT_MODE="$TRAIN_STDOUT_MODE" \
TRAIN_STDOUT_FILTER="$TRAIN_STDOUT_FILTER" \
bash "$REPO/deploy/v2/train.sh" 2>&1 | tee "$WORK/logs/train.live.log"
