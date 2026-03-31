#!/bin/bash
# AlphaAgentEvo v2 — Kaggle-friendly Verl launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

REPO_ROOT="$(v2_repo_root)"
WORK="${WORK:-$(v2_default_work_dir)}"
VERL="$WORK/verl"
DATA="$WORK/data"
LOGS="$WORK/logs"
ENV_NAME="${ENV_NAME:-verl041}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B-MLX-bf16}"
API_BASE_URL="${API_BASE_URL:-http://localhost:8002}"
HEALTH_URL="${API_BASE_URL%/}/health"
export ALPHAEVO_BACKTEST_API_URL="${ALPHAEVO_BACKTEST_API_URL:-$API_BASE_URL/backtest}"
MODEL_TAG_RAW="${MODEL##*/}"
MODEL_TAG="$(echo "$MODEL_TAG_RAW" | tr '/: ' '---' | tr -cd '[:alnum:]_.-')"
if [ -z "$MODEL_TAG" ]; then
    MODEL_TAG="model"
fi

USE_LORA="${USE_LORA:-0}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

if [ "$USE_LORA" = "1" ]; then
    ACTOR_LORA_RANK="$LORA_RANK"
    CRITIC_LORA_RANK="$LORA_RANK"
    LORA_STATUS="enabled (rank=$LORA_RANK alpha=$LORA_ALPHA target_modules=$LORA_TARGET_MODULES)"
    DEFAULT_EXPERIMENT_NAME="${MODEL_TAG}-verl-gpu\${N_GPUS}-lora"
else
    ACTOR_LORA_RANK=0
    CRITIC_LORA_RANK=0
    LORA_STATUS="disabled"
    DEFAULT_EXPERIMENT_NAME="${MODEL_TAG}-verl-gpu\${N_GPUS}-fullft"
fi

v2_activate_env "$WORK"

if [ ! -d "$VERL" ]; then
    echo "ERROR: $VERL not found. Run $SCRIPT_DIR/setup.sh first." >&2
    exit 1
fi

if echo "$MODEL" | grep -qi "fp8"; then
    echo "WARNING: MODEL=$MODEL appears to be FP8. Current verl+sglang runtime may crash during validation weight sync with FP8 Qwen3."
    echo "         Recommended for stability: use a BF16 model variant (e.g. Qwen/Qwen3-0.6B-MLX-bf16)."
fi

mkdir -p "$DATA"
echo "[train-entry] refreshing normalized datasets"
python "$SCRIPT_DIR/prepare_dataset.py" --input-dir "$SCRIPT_DIR" --output-dir "$DATA" --repo-root "$REPO_ROOT" --splits train,val

for f in \
    "$DATA/train.parquet" \
    "$DATA/val.parquet" \
    "$VERL/verl/tools/factor_tool.py" \
    "$VERL/verl/utils/reward_score/factor.py"; do
    [ -f "$f" ] || { echo "ERROR: Missing required file $f" >&2; exit 1; }
done

echo "[hotfix] Syncing runtime integrations into $VERL"
cp "$SCRIPT_DIR/factor_tool.py" "$VERL/verl/tools/factor_tool.py"
mkdir -p "$VERL/examples/sglang_multiturn/config/tool_config"
cp "$SCRIPT_DIR/factor_tool_config.yaml" "$VERL/examples/sglang_multiturn/config/tool_config/factor_tool_config.yaml"
mkdir -p "$VERL/verl/utils/reward_score"
cp "$SCRIPT_DIR/factor_reward.py" "$VERL/verl/utils/reward_score/factor.py"
python "$SCRIPT_DIR/patch_verl.py" --verl-dir "$VERL"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. This launcher expects a CUDA GPU host." >&2
    exit 1
fi

N_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [ "$N_GPUS" -lt 1 ]; then
    echo "ERROR: No GPUs detected." >&2
    exit 1
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-3}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-3}"
MAX_TOOL_CALLS_PER_TURN="${MAX_TOOL_CALLS_PER_TURN:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-40}"
TEST_FREQ="${TEST_FREQ:-40}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
ROLLOUT_DISABLE_LOG_STATS="${ROLLOUT_DISABLE_LOG_STATS:-False}"
ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE="${ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE:-False}"
ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-True}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-$ROLLOUT_N}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
ACTOR_CHECKPOINT_SAVE_CONTENTS="${ACTOR_CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"extra\",\"hf_model\"]}"
ACTOR_CHECKPOINT_LOAD_CONTENTS="${ACTOR_CHECKPOINT_LOAD_CONTENTS:-[\"model\",\"extra\"]}"
CRITIC_CHECKPOINT_SAVE_CONTENTS="${CRITIC_CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"extra\"]}"
CRITIC_CHECKPOINT_LOAD_CONTENTS="${CRITIC_CHECKPOINT_LOAD_CONTENTS:-[\"model\",\"extra\"]}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-4096}"
ROLLOUT_MAX_SEQS="${ROLLOUT_MAX_SEQS:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}"
FORWARD_MAX_TOKEN_LEN_PER_GPU="${FORWARD_MAX_TOKEN_LEN_PER_GPU:-4096}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-4096}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
CRITIC_FORWARD_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_FORWARD_MICRO_BATCH_SIZE_PER_GPU:-1}"
CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.08}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
KL_COEF="${KL_COEF:-0.001}"
MODEL_ATTN_IMPL="${MODEL_ATTN_IMPL:-sdpa}"
TRAIN_HEARTBEAT_SEC="${TRAIN_HEARTBEAT_SEC:-30}"
ALPHAEVO_BACKTEST_TIMEOUT_SEC="${ALPHAEVO_BACKTEST_TIMEOUT_SEC:-120}"
TRAIN_STDOUT_MODE="${TRAIN_STDOUT_MODE:-pretty}" # raw|pretty|filtered|off
TRAIN_STDOUT_FILTER="${TRAIN_STDOUT_FILTER:-Starting AlphaAgentEvo|heartbeat|step:|critic/rewards/mean|val-core/alphaagentevo/reward/mean@3|val-paper/|\\[FactorTool\\] backtest success|\\[factor-live\\]|\\[response-live\\]|TRAIN SUMMARY|ERROR|RayTaskError}"
eval "DEFAULT_EXPERIMENT_NAME_EXPANDED=\"$DEFAULT_EXPERIMENT_NAME\""
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME_EXPANDED}"

TRAIN_EXTRA_ARGS=("$@")
if [ -n "$RESUME_FROM_PATH" ]; then
    TRAIN_EXTRA_ARGS+=(trainer.resume_from_path="$RESUME_FROM_PATH")
fi

if [ "$MAX_MODEL_LEN" -lt $((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) ]; then
    echo "ERROR: Invalid token budget." >&2
    echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN must be >= MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH = $((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" >&2
    exit 1
fi

if [ "$TEST_FREQ" -gt "$TOTAL_STEPS" ] && [ "$VAL_BEFORE_TRAIN" != "True" ]; then
    echo "WARNING: TEST_FREQ ($TEST_FREQ) > TOTAL_STEPS ($TOTAL_STEPS) and VAL_BEFORE_TRAIN=False; validation may not run during training."
fi

case "$TRAIN_STDOUT_MODE" in
    raw|pretty|filtered|off) ;;
    *)
        echo "ERROR: TRAIN_STDOUT_MODE must be one of: raw, pretty, filtered, off. Got: $TRAIN_STDOUT_MODE" >&2
        exit 1
        ;;
esac

if [ "$ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE" = "True" ]; then
    echo "WARNING: ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE=True can cause Qwen3 to emit visible <think> traces or prose instead of strict <tool_call> blocks."
    echo "         Recommended for this repo's paper-style tool trajectories: set ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE=False."
fi

mkdir -p "$LOGS"
TRAIN_LIVE_LOG="${TRAIN_LIVE_LOG:-$LOGS/train.live.log}"
TRAIN_PRETTY_LOG="${TRAIN_PRETTY_LOG:-$LOGS/train.pretty.log}"
rm -f "$TRAIN_PRETTY_LOG"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$WORK:$VERL:$WORK/backtest:$WORK/expression_manager:$REPO_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export NCCL_TIMEOUT=7200
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCH_NCCL_TIMEOUT=7200
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPUS-1)))}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export ALPHAEVO_BACKTEST_TIMEOUT_SEC
ulimit -n 65535

if [ "$RUN_PREFLIGHT" = "1" ]; then
    echo "[train-entry] runtime preflight start"
    python "$SCRIPT_DIR/runtime_preflight.py" --work-dir "$WORK" --skip-pip-check
    echo "[train-entry] runtime preflight done"
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
    echo "  Check $LOGS/api.log or start it manually: python $REPO_ROOT/deploy/api_server_verl.py" >&2
    exit 1
}

echo "============================================================"
echo "Starting AlphaAgentEvo v2 GRPO Training"
echo "  Repo:            $REPO_ROOT"
echo "  Work:            $WORK"
echo "  Model:           $MODEL"
echo "  GPUs:            $N_GPUS"
echo "  Batch:           $TRAIN_BATCH_SIZE"
echo "  Val batch:       $VAL_BATCH_SIZE"
echo "  PPO mini batch:  $PPO_MINI_BATCH_SIZE"
echo "  Rollouts:        $ROLLOUT_N"
echo "  Val rollouts:    $VAL_ROLLOUT_N"
echo "  Max turns:       $MAX_ASSISTANT_TURNS"
echo "  Max tool calls:  $MAX_TOOL_CALLS_PER_TURN (prompt contract)"
echo "  Chunked prefill: $ROLLOUT_ENABLE_CHUNKED_PREFILL"
echo "  Rollout max seqs:$ROLLOUT_MAX_SEQS"
echo "  GPU mem util:    $GPU_MEMORY_UTILIZATION"
echo "  Rollout stats:   disable_log_stats=$ROLLOUT_DISABLE_LOG_STATS"
echo "  Actor ckpt save: $ACTOR_CHECKPOINT_SAVE_CONTENTS"
echo "  Critic ckpt save:$CRITIC_CHECKPOINT_SAVE_CONTENTS"
echo "  LoRA:            $LORA_STATUS"
echo "  API:             $ALPHAEVO_BACKTEST_API_URL"
echo "  Save freq:       $SAVE_FREQ"
echo "  Test freq:       $TEST_FREQ"
echo "  Val before train:$VAL_BEFORE_TRAIN"
echo "  Resume mode:     $RESUME_MODE"
echo "  Inference chat:  $ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE"
echo "  Stdout mode:     $TRAIN_STDOUT_MODE"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

cd "$VERL"

watchdog() {
    local target_pid="$1"
    local interval="${TRAIN_HEARTBEAT_SEC:-30}"
    local last_size=-1
    local last_progress_events=-1
    local stagnant_rounds=0
    local stall_min_bytes="${TRAIN_STALL_MIN_BYTES:-2048}"
    while kill -0 "$target_pid" >/dev/null 2>&1; do
        if [ ! -d "$LOGS" ]; then
            echo "[heartbeat] waiting for logs dir: $LOGS"
            sleep "$interval"
            continue
        fi
        local train_log=""
        if [ -f "$LOGS/train.live.log" ]; then
            train_log="$LOGS/train.live.log"
        elif [ -f "$LOGS/train.log" ]; then
            train_log="$LOGS/train.log"
        fi

        local size="NA"
        local progress_events="NA"
        local phase="NA"
        local size_delta=0
        local warmup_progress=0
        if [ -n "$train_log" ]; then
            size="$(wc -c <"$train_log" | tr -d ' ')"
            # Track real progress markers, not raw log size (heartbeat itself grows the log).
            progress_events="$(grep -Ec "step:|\\[FactorTool\\] backtest success|Training Progress:[[:space:]]+[1-9]|Error executing job|RayTaskError|AssertionError|val-core/alphaagentevo/reward/mean@3|val-paper/|Capturing batches" "$train_log" || true)"
            phase="$(grep -v '^\[heartbeat\]' "$train_log" | tail -n 1 | cut -c1-140 || true)"
            if [ "$last_size" -ge 0 ] 2>/dev/null; then
                size_delta=$((size - last_size))
            else
                size_delta=0
            fi
            if echo "${phase:-}" | grep -Eq "Capturing batches|%\\|"; then
                warmup_progress=1
            fi
            if [ "$progress_events" != "$last_progress_events" ] || [ "$size_delta" -ge "$stall_min_bytes" ] || [ "$warmup_progress" -eq 1 ]; then
                stagnant_rounds=0
                last_progress_events="$progress_events"
            else
                stagnant_rounds=$((stagnant_rounds + 1))
            fi
            last_size="$size"
        fi

        local gpu_line
        gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)"
        local api_status="down"
        if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
            api_status="up"
        fi

        echo "[heartbeat] $(date '+%Y-%m-%d %H:%M:%S') pid=$target_pid log_bytes=$size delta_bytes=$size_delta progress_events=$progress_events stagnant_rounds=$stagnant_rounds api=$api_status gpu(util,used,total)=${gpu_line:-NA} phase=${phase:-NA}"
        if [ "$stagnant_rounds" -ge 5 ]; then
            echo "[heartbeat] no progress markers for $((stagnant_rounds * interval))s; may be stalled or in long compile/capture. Inspect /tmp/ray/session_latest/logs/worker-*.err"
        fi
        sleep "$interval"
    done
}

dump_failure_context() {
    echo "=================== TRAIN FAILURE CONTEXT ==================="
    echo "--- tail $LOGS/api.log ---"
    tail -n 120 "$LOGS/api.log" 2>/dev/null || true
    echo "--- recent ray worker stderr ---"
    local f
    for f in $(ls -t /tmp/ray/session_latest/logs/worker-*.err 2>/dev/null | head -n 3); do
        echo ">>> $f"
        tail -n 120 "$f" || true
    done
    if [ -x "$SCRIPT_DIR/check_train_error.sh" ]; then
        WORK="$WORK" LOGS_DIR="$LOGS" RAY_LOG_DIR="/tmp/ray/session_latest/logs" "$SCRIPT_DIR/check_train_error.sh" || true
    fi
    echo "============================================================="
}

summarize_train_progress() {
    local log_file="$TRAIN_LIVE_LOG"
    [ -f "$log_file" ] || return 0
    echo "=================== TRAIN SUMMARY ==================="
    echo "backtest success count:"
    grep -c "\\[FactorTool\\] backtest success" "$log_file" || true
    echo
    echo "validation points logged:"
    grep -c "val-core/alphaagentevo/reward/mean@3" "$log_file" || true
    echo
    echo "recent step rewards:"
    grep -E "step:[0-9]+ .*critic/rewards/mean" "$log_file" | tail -n 10 || true
    echo
    echo "recent paper-style validation metrics:"
    grep -E "val-paper/.*/(vr|pass@3|pass@5|beat_rate|best_metric_mean)" "$log_file" | tail -n 10 || true
    if [ -f "$TRAIN_PRETTY_LOG" ]; then
        echo
        echo "recent pretty log:"
        tail -n 20 "$TRAIN_PRETTY_LOG" || true
    fi
    echo "====================================================="
}

is_benign_post_train_failure() {
    local log_file="$TRAIN_LIVE_LOG"
    [ -f "$log_file" ] || return 1

    grep -Eq "Training Progress: 100%|global_step_${TOTAL_STEPS}" "$log_file" || return 1
    grep -q "Final validation metrics:" "$log_file" || return 1
    grep -Eq "Saved hf_model|Saved model to .*global_step_${TOTAL_STEPS}" "$log_file" || return 1
    grep -Eq "RuntimeError: DataLoader worker .* is killed by signal: Killed" "$log_file" || return 1
    return 0
}

echo "[train-entry] launching trainer process"
echo "[train-entry] live_log=$TRAIN_LIVE_LOG pretty_log=$TRAIN_PRETTY_LOG stdout_mode=$TRAIN_STDOUT_MODE heartbeat_sec=$TRAIN_HEARTBEAT_SEC"
(
    set -o pipefail
    stdbuf -oL -eL python -u -m verl.trainer.main_ppo \
    --config-path="$VERL/examples/sglang_multiturn/config" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.shuffle_train_dataloader=False \
    actor_rollout_ref.model.path="$MODEL" \
    +actor_rollout_ref.model.override_config._attn_implementation="$MODEL_ATTN_IMPL" \
    actor_rollout_ref.model.lora_rank="$ACTOR_LORA_RANK" \
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
    actor_rollout_ref.model.target_modules="$LORA_TARGET_MODULES" \
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${LR_WARMUP_RATIO:-0.1}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.checkpoint.save_contents="$ACTOR_CHECKPOINT_SAVE_CONTENTS" \
    actor_rollout_ref.actor.checkpoint.load_contents="$ACTOR_CHECKPOINT_LOAD_CONTENTS" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.00 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_SEQS" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
    actor_rollout_ref.rollout.enable_chunked_prefill="$ROLLOUT_ENABLE_CHUNKED_PREFILL" \
    actor_rollout_ref.rollout.disable_log_stats="$ROLLOUT_DISABLE_LOG_STATS" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.do_sample="$ROLLOUT_DO_SAMPLE" \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_ROLLOUT_N" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template="$ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE" \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path="$MODEL" \
    critic.model.tokenizer_path="$MODEL" \
    +critic.model.override_config._attn_implementation="$MODEL_ATTN_IMPL" \
    critic.model.use_remove_padding=False \
    critic.checkpoint.save_contents="$CRITIC_CHECKPOINT_SAVE_CONTENTS" \
    critic.checkpoint.load_contents="$CRITIC_CHECKPOINT_LOAD_CONTENTS" \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.lora_rank="$CRITIC_LORA_RANK" \
    critic.model.lora_alpha="$LORA_ALPHA" \
    critic.model.target_modules="$LORA_TARGET_MODULES" \
    critic.forward_max_token_len_per_gpu="$FORWARD_MAX_TOKEN_LEN_PER_GPU" \
    critic.forward_micro_batch_size_per_gpu="$CRITIC_FORWARD_MICRO_BATCH_SIZE_PER_GPU" \
    critic.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    critic.ppo_micro_batch_size_per_gpu="$CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU" \
    custom_reward_function.path="$VERL/verl/utils/reward_score/factor.py" \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.resume_mode="$RESUME_MODE" \
    'trainer.logger=["console","tensorboard"]' \
    trainer.project_name='alphaagentevo-v2' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/val.parquet" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$VERL/examples/sglang_multiturn/config/tool_config/factor_tool_config.yaml" \
    trainer.total_training_steps="$TOTAL_STEPS" \
    "${TRAIN_EXTRA_ARGS[@]}" 2>&1 \
    | tee "$TRAIN_LIVE_LOG" \
    | {
        if [ "$TRAIN_STDOUT_MODE" = "off" ]; then
            cat >/dev/null
        elif [ "$TRAIN_STDOUT_MODE" = "pretty" ]; then
            python -u "$SCRIPT_DIR/train_stream_pretty.py" \
                --total-steps "$TOTAL_STEPS" \
                --pretty-log "$TRAIN_PRETTY_LOG"
        elif [ "$TRAIN_STDOUT_MODE" = "filtered" ]; then
            grep --line-buffered -E "$TRAIN_STDOUT_FILTER" || true
        else
            cat
        fi
    }
) &
TRAIN_PID=$!
echo "[train-entry] trainer pid=$TRAIN_PID"

if [ "${TRAIN_HEARTBEAT_SEC:-0}" -gt 0 ]; then
    watchdog "$TRAIN_PID"
fi

set +e
wait "$TRAIN_PID"
TRAIN_RC=$?
set -e

summarize_train_progress

if [ "$TRAIN_RC" -ne 0 ] && is_benign_post_train_failure; then
    echo "[train-exit] detected benign post-train DataLoader worker failure after successful completion; normalizing exit code to 0"
    TRAIN_RC=0
fi

if [ "$TRAIN_RC" -ne 0 ]; then
    dump_failure_context
fi

exit "$TRAIN_RC"
