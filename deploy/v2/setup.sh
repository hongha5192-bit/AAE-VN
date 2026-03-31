#!/bin/bash
# AlphaAgentEvo v2 — portable setup for Kaggle / single-H100 / Runpod
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

REPO_ROOT="$(v2_repo_root)"
WORK="${WORK:-$(v2_default_work_dir)}"
VERL="$WORK/verl"
DATA="$WORK/data"
LOGS="$WORK/logs"
CHECKPOINTS="$WORK/checkpoints"
SETUP_REV="2026-03-26-mamba-lock-preflight-r4"
ENV_NAME="${ENV_NAME:-verl041}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERL_COMMIT="${VERL_COMMIT:-8d9e350ea58c7ad4b50dd14d9dcb50577242c55f}"
MICROMAMBA_DIR="$WORK/micromamba"
MICROMAMBA_BIN="$MICROMAMBA_DIR/bin/micromamba"
MAMBA_ROOT_PREFIX="$MICROMAMBA_DIR/root-prefix"
CONSTRAINTS_FILE="$SCRIPT_DIR/constraints.txt"
SETUP_DATA_SPLITS="${SETUP_DATA_SPLITS:-train,val}"
SETUP_STAMP="$WORK/.setup_v2_stamp"

echo "============================================================"
echo "AlphaAgentEvo v2 — Setup"
echo "  Repo: $REPO_ROOT"
echo "  Work: $WORK"
echo "  Env:  $ENV_NAME"
echo "  Rev:  $SETUP_REV"
echo "============================================================"

mkdir -p "$DATA" "$LOGS" "$CHECKPOINTS" "$MICROMAMBA_DIR"

if [ ! -f "$REPO_ROOT/data/daily_pv.h5" ]; then
    echo "ERROR: Missing repo-local data file: $REPO_ROOT/data/daily_pv.h5" >&2
    echo "  This Kaggle bundle is expected to include daily_pv.h5 inside the repo." >&2
    exit 1
fi

if [ ! -f "$REPO_ROOT/backtest/data/daily_pv.h5" ]; then
    echo "ERROR: Missing backtest data file: $REPO_ROOT/backtest/data/daily_pv.h5" >&2
    exit 1
fi

echo "[1/7] Create or activate Python 3.10 environment"
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
    if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        echo "ERROR: conda found but conda.sh is missing." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    if [ "${RECREATE_ENV:-0}" = "1" ]; then
        conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
    fi
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        conda create -n "$ENV_NAME" python=3.10 -y
    fi
    conda activate "$ENV_NAME"
else
    if [ ! -x "$MICROMAMBA_BIN" ]; then
        echo "  Bootstrapping micromamba into $MICROMAMBA_DIR"
        rm -rf "$MICROMAMBA_DIR/bin"
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MICROMAMBA_DIR" bin/micromamba
    fi
    export MAMBA_ROOT_PREFIX
    eval "$("$MICROMAMBA_BIN" shell hook -s bash -r "$MAMBA_ROOT_PREFIX")"
    if [ "${RECREATE_ENV:-0}" = "1" ]; then
        micromamba env remove -y -n "$ENV_NAME" >/dev/null 2>&1 || true
    fi
    if [ ! -d "$MAMBA_ROOT_PREFIX/envs/$ENV_NAME" ]; then
        micromamba create -y -n "$ENV_NAME" python=3.10 pip
    fi
    micromamba activate "$ENV_NAME"
fi

PYTHON_VERSION="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "ERROR: Python 3.10 is required for this v2 stack. Current: $PYTHON_VERSION" >&2
    exit 1
fi
echo "Python: $(python --version)"
echo "Executable: $(command -v python)"

STAMP_PAYLOAD="$(cat <<EOF
rev=$SETUP_REV
env=$ENV_NAME
verl_commit=$VERL_COMMIT
data_splits=$SETUP_DATA_SPLITS
EOF
)"

if [ "${FORCE_SETUP:-0}" != "1" ] && [ -f "$SETUP_STAMP" ]; then
    EXISTING_STAMP="$(cat "$SETUP_STAMP" 2>/dev/null || true)"
    if [ "$EXISTING_STAMP" = "$STAMP_PAYLOAD" ] \
        && [ -d "$VERL" ] \
        && [ -f "$DATA/train.parquet" ] \
        && [ -f "$DATA/val.parquet" ]; then
        echo "[cache] Matching setup stamp found; running quick preflight"
        if python "$SCRIPT_DIR/runtime_preflight.py" --work-dir "$WORK" --skip-pip-check; then
            echo "[cache] Runtime already prepared. Skipping full setup."
            echo "Tip: set FORCE_SETUP=1 if you want to reinstall anyway."
            exit 0
        fi
        echo "[cache] Quick preflight failed; continuing with full setup"
    fi
fi

echo "[2/7] Clone Verl v0.4.1"
rm -rf "$VERL"
git clone --branch v0.4.1 --depth 1 https://github.com/verl-project/verl.git "$VERL"
cd "$VERL"
git fetch --depth 1 origin "$VERL_COMMIT" >/dev/null 2>&1 || true
git checkout "$VERL_COMMIT"

echo "[3/7] Install pinned packages"
echo "  [3/7a] Upgrade pip/setuptools/wheel"
python -m pip install --upgrade pip "setuptools<81" wheel
echo "  [3/7b] Remove conflicting runtime packages"
python -m pip uninstall -y sglang sgl-kernel flashinfer-python verl flash_attn flash-attn 2>/dev/null || true
echo "  [3/7c] Install core runtime packages"
python -m pip install -c "$CONSTRAINTS_FILE" "torch==2.6.0" "tensordict==0.6.2" "sglang[srt,openai]==0.4.6.post5" "torch-memory-saver>=0.0.5"
echo "  [3/7d] Install local Verl editable package"
python -m pip install -c "$CONSTRAINTS_FILE" -e .
echo "  [3/7e] Install service/data dependencies"
python -m pip install -c "$CONSTRAINTS_FILE" fastapi uvicorn requests pyarrow tables h5py jmespath joblib scipy pyparsing tensorboard pandas
python -m pip check
python -m pip freeze > "$LOGS/pip-freeze.txt"

if ! python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("pkg_resources") else 1)
PY
then
    echo "  pkg_resources missing; reinstalling setuptools from pip wheel"
    python -m pip install --force-reinstall --no-deps "setuptools<81"
fi

echo "[4/7] Install system extras when available"
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y -qq libnuma-dev >/dev/null 2>&1 || true
fi
command -v ldconfig >/dev/null 2>&1 && ldconfig || true

echo "[5/7] Copy AlphaAgentEvo v2 integrations into Verl"
cp "$SCRIPT_DIR/factor_tool.py" "$VERL/verl/tools/factor_tool.py"
mkdir -p "$VERL/examples/sglang_multiturn/config/tool_config"
cp "$SCRIPT_DIR/factor_tool_config.yaml" "$VERL/examples/sglang_multiturn/config/tool_config/factor_tool_config.yaml"
mkdir -p "$VERL/verl/utils/reward_score"
cp "$SCRIPT_DIR/factor_reward.py" "$VERL/verl/utils/reward_score/factor.py"
python "$SCRIPT_DIR/patch_verl.py" --verl-dir "$VERL"

echo "[6/7] Prepare normalized datasets + runtime links"
echo "  [6/7a] Building dataset splits: $SETUP_DATA_SPLITS"
python "$SCRIPT_DIR/prepare_dataset.py" --input-dir "$SCRIPT_DIR" --output-dir "$DATA" --repo-root "$REPO_ROOT" --splits "$SETUP_DATA_SPLITS"
echo "  [6/7b] Refresh runtime symlinks"
rm -rf "$WORK/backtest" "$WORK/expression_manager"
ln -s "$REPO_ROOT/backtest" "$WORK/backtest"
ln -s "$REPO_ROOT/expression_manager" "$WORK/expression_manager"

echo "[7/7] Verify install"
VERIFY_FILES=(
    "$DATA/train.parquet"
    "$DATA/val.parquet"
    "$VERL/verl/tools/factor_tool.py"
    "$VERL/verl/workers/reward_manager/naive.py"
    "$VERL/verl/trainer/ppo/ray_trainer.py"
    "$VERL/verl/trainer/ppo/metric_utils.py"
    "$VERL/verl/utils/reward_score/factor.py"
)

if [[ ",$SETUP_DATA_SPLITS," == *",test,"* ]]; then
    VERIFY_FILES+=("$DATA/test.parquet")
fi

for f in "${VERIFY_FILES[@]}"; do
    [ -f "$f" ] && echo "  OK: $f" || { echo "  MISSING: $f"; exit 1; }
done

python "$SCRIPT_DIR/runtime_preflight.py" --work-dir "$WORK"

python - <<PY
import pandas as pd
import torch
import sglang
import tensordict
import verl
from verl.tools.factor_tool import FactorTool
print(f"torch {torch.__version__}")
print(f"sglang {sglang.__version__}")
print(f"tensordict {tensordict.__version__}")
print("verl OK")
print("factor_tool import OK")
df = pd.read_parquet("$DATA/train.parquet")
assert "evaluate_factor" in df.iloc[0]["tools_kwargs"], df.iloc[0]["tools_kwargs"]
print("dataset OK")
print("ALL OK")
PY

printf '%s\n' "$STAMP_PAYLOAD" > "$SETUP_STAMP"

echo
echo "============================================================"
echo "Setup complete."
echo "Next:"
echo "  1. bash $SCRIPT_DIR/train.sh"
echo "  2. tail -f $LOGS/train.log"
echo "  3. Re-run setup quickly with cached stamp, or set FORCE_SETUP=1 to reinstall"
echo "============================================================"
