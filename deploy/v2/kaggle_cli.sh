#!/bin/bash
# Kaggle CLI automation for AlphaAgentEvo v2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

KAGGLE_BIN="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
KAGGLE_TOKEN_FILE="$KAGGLE_CONFIG_DIR/kaggle.json"

KAGGLE_DATASET_PATH="${KAGGLE_DATASET_PATH:-$REPO_ROOT}"
KAGGLE_DATASET_ID="${KAGGLE_DATASET_ID:-giaphlm/aae-new}"
KAGGLE_KERNEL_PATH="${KAGGLE_KERNEL_PATH:-$REPO_ROOT/deploy/v2/kaggle_kernel}"
KAGGLE_KERNEL_ID="${KAGGLE_KERNEL_ID:-giaphlm/aae-new-train}"
KAGGLE_OUTPUT_DIR="${KAGGLE_OUTPUT_DIR:-$REPO_ROOT/.kaggle-output}"
KAGGLE_COMPETITION_SOURCE="${KAGGLE_COMPETITION_SOURCE:-ai-mathematical-olympiad-progress-prize-3}"
KAGGLE_ENABLE_INTERNET="${KAGGLE_ENABLE_INTERNET:-true}"
KAGGLE_ENABLE_GPU="${KAGGLE_ENABLE_GPU:-true}"
KAGGLE_MACHINE_SHAPE="${KAGGLE_MACHINE_SHAPE-NvidiaT4}"
KAGGLE_KERNEL_TITLE="${KAGGLE_KERNEL_TITLE:-}"
KAGGLE_NOTEBOOK_TEMPLATE="${KAGGLE_NOTEBOOK_TEMPLATE:-alphaagentevo.ipynb}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_kaggle() {
    [ -x "$KAGGLE_BIN" ] || die "Kaggle CLI not found at $KAGGLE_BIN"
}

require_auth() {
    [ -f "$KAGGLE_TOKEN_FILE" ] || die "Missing token file: $KAGGLE_TOKEN_FILE"
    # Kaggle CLI v2 prefers access-token auth for tokens like KGAT_...
    local detected_key=""
    detected_key="$(python3 - <<PY
import json
from pathlib import Path
p = Path("$KAGGLE_TOKEN_FILE")
try:
    data = json.loads(p.read_text())
    print(data.get("key",""))
except Exception:
    print("")
PY
)"
    if [[ "$detected_key" == KGAT_* ]]; then
        export KAGGLE_API_TOKEN="$detected_key"
    fi
}

cmd_doctor() {
    require_kaggle
    echo "Kaggle CLI: $($KAGGLE_BIN --version)"
    echo "Config dir: $KAGGLE_CONFIG_DIR"
    if [ -f "$KAGGLE_TOKEN_FILE" ]; then
        perms="$(stat -f '%Lp' "$KAGGLE_TOKEN_FILE" 2>/dev/null || stat -c '%a' "$KAGGLE_TOKEN_FILE")"
        echo "Token file: $KAGGLE_TOKEN_FILE (perm=$perms)"
    else
        echo "Token file missing: $KAGGLE_TOKEN_FILE"
    fi

    if [ -f "$REPO_ROOT/dataset-metadata.json" ]; then
        echo "Dataset metadata: $REPO_ROOT/dataset-metadata.json"
    else
        echo "Dataset metadata missing: $REPO_ROOT/dataset-metadata.json"
    fi
    if [ -f "$KAGGLE_KERNEL_PATH/kernel-metadata.json" ]; then
        echo "Kernel metadata: $KAGGLE_KERNEL_PATH/kernel-metadata.json"
        python3 - <<PY
import json
p = "$KAGGLE_KERNEL_PATH/kernel-metadata.json"
d = json.load(open(p))
print("  id:", d.get("id"))
print("  dataset_sources:", d.get("dataset_sources"))
print("  competition_sources:", d.get("competition_sources"))
print("  enable_gpu:", d.get("enable_gpu"))
print("  enable_internet:", d.get("enable_internet"))
PY
    else
        echo "Kernel metadata missing: $KAGGLE_KERNEL_PATH/kernel-metadata.json"
    fi

    if [ -f "$KAGGLE_TOKEN_FILE" ]; then
        require_auth
        echo "Auth smoke test (kernels list --mine):"
        "$KAGGLE_BIN" kernels list --mine >/dev/null && echo "  OK" || echo "  FAILED"

        echo "Dataset status ($KAGGLE_DATASET_ID):"
        ds_out="$("$KAGGLE_BIN" datasets status "$KAGGLE_DATASET_ID" 2>&1 || true)"
        if echo "$ds_out" | grep -qiE "401|403|unauthorized|forbidden"; then
            echo "  WARN: $ds_out"
        else
            echo "  $ds_out"
        fi

        echo "Kernel status ($KAGGLE_KERNEL_ID):"
        ks_out="$("$KAGGLE_BIN" kernels status "$KAGGLE_KERNEL_ID" 2>&1 || true)"
        if echo "$ks_out" | grep -qiE "403|404|forbidden|not found"; then
            echo "  WARN: kernel may not exist yet or access is restricted: $ks_out"
        else
            echo "  $ks_out"
        fi
    fi
}

cmd_kernel_config() {
    [ -f "$KAGGLE_KERNEL_PATH/kernel-metadata.json" ] || die "kernel-metadata.json not found in $KAGGLE_KERNEL_PATH"
    python3 - <<PY
import json
import shutil
from pathlib import Path

p = Path("$KAGGLE_KERNEL_PATH/kernel-metadata.json")
d = json.loads(p.read_text())
def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

d["id"] = "${KAGGLE_KERNEL_ID}"
slug = "${KAGGLE_KERNEL_ID}".split("/")[-1].strip()
d["title"] = "${KAGGLE_KERNEL_TITLE}" or slug.replace("-", " ").title()
d["kernel_type"] = "notebook"
code_file = f"{slug}.ipynb"
template = Path("$KAGGLE_KERNEL_PATH") / "${KAGGLE_NOTEBOOK_TEMPLATE}"
target = Path("$KAGGLE_KERNEL_PATH") / code_file
if not target.exists():
    if not template.exists():
        raise FileNotFoundError(f"Notebook template not found: {template}")
    shutil.copyfile(template, target)
d["code_file"] = code_file
d["dataset_sources"] = ["${KAGGLE_DATASET_ID}"]
d["enable_internet"] = _to_bool("${KAGGLE_ENABLE_INTERNET}")
d["enable_gpu"] = _to_bool("${KAGGLE_ENABLE_GPU}")

comp = "${KAGGLE_COMPETITION_SOURCE}".strip()
if comp:
    d["competition_sources"] = [comp]
else:
    d["competition_sources"] = []

machine_shape = "${KAGGLE_MACHINE_SHAPE}".strip()
if machine_shape:
    d["machine_shape"] = machine_shape
else:
    d.pop("machine_shape", None)

p.write_text(json.dumps(d, indent=2) + "\\n")
print(f"Updated {p}")
print(json.dumps(d, indent=2))
PY
}

cmd_dataset_version() {
    require_kaggle
    require_auth
    [ -f "$KAGGLE_DATASET_PATH/dataset-metadata.json" ] || die "dataset-metadata.json not found in $KAGGLE_DATASET_PATH"
    local msg="${1:-auto update $(date '+%Y-%m-%d %H:%M:%S')}"
    echo "Versioning dataset $KAGGLE_DATASET_ID from $KAGGLE_DATASET_PATH"
    "$KAGGLE_BIN" datasets version -p "$KAGGLE_DATASET_PATH" -m "$msg" -r tar
}

cmd_kernel_push() {
    require_kaggle
    require_auth
    [ -f "$KAGGLE_KERNEL_PATH/kernel-metadata.json" ] || die "kernel-metadata.json not found in $KAGGLE_KERNEL_PATH"
    echo "Pushing kernel $KAGGLE_KERNEL_ID from $KAGGLE_KERNEL_PATH"
    set +e
    out="$("$KAGGLE_BIN" kernels push -p "$KAGGLE_KERNEL_PATH" 2>&1)"
    rc=$?
    set -e
    echo "$out"
    if [ $rc -ne 0 ]; then
        die "kaggle kernels push failed (exit=$rc)"
    fi
    if echo "$out" | grep -qi "Notebook not found"; then
        die "kernel ref not writable via API. Try: KAGGLE_KERNEL_ID=giaphlm/aae-new-train bash deploy/v2/kaggle_cli.sh kernel-config && bash deploy/v2/kaggle_cli.sh kernel-push"
    fi
    if echo "$out" | grep -qiE "kernel push error|error:"; then
        die "kaggle kernels push reported an error"
    fi
}

cmd_kernel_status() {
    require_kaggle
    require_auth
    "$KAGGLE_BIN" kernels status "$KAGGLE_KERNEL_ID"
}

cmd_kernel_watch() {
    require_kaggle
    require_auth
    local timeout_sec="${1:-7200}"
    local interval_sec="${2:-30}"
    local elapsed=0
    while [ "$elapsed" -lt "$timeout_sec" ]; do
        out="$("$KAGGLE_BIN" kernels status "$KAGGLE_KERNEL_ID" 2>&1 || true)"
        ts="$(date '+%Y-%m-%d %H:%M:%S')"
        echo "[$ts] $out"
        if echo "$out" | tr '[:upper:]' '[:lower:]' | grep -Eq "complete|failed|error|cancel"; then
            return 0
        fi
        sleep "$interval_sec"
        elapsed=$((elapsed + interval_sec))
    done
    die "Timeout while waiting for kernel status"
}

cmd_kernel_output() {
    require_kaggle
    require_auth
    mkdir -p "$KAGGLE_OUTPUT_DIR"
    "$KAGGLE_BIN" kernels output "$KAGGLE_KERNEL_ID" -p "$KAGGLE_OUTPUT_DIR" -o
    echo "Downloaded outputs to: $KAGGLE_OUTPUT_DIR"
}

cmd_feedback() {
    local log_file="${1:-$KAGGLE_OUTPUT_DIR/train.log}"
    python3 "$SCRIPT_DIR/autofeedback.py" --train-log "$log_file"
}

cmd_full_cycle() {
    local msg="${1:-auto full cycle $(date '+%Y-%m-%d %H:%M:%S')}"
    cmd_kernel_config
    cmd_dataset_version "$msg"
    cmd_kernel_push
    cmd_kernel_watch 7200 30
    cmd_kernel_output
    cmd_feedback "$KAGGLE_OUTPUT_DIR/train.log" || true
}

usage() {
    cat <<'EOF'
Usage:
  bash deploy/v2/kaggle_cli.sh doctor
  bash deploy/v2/kaggle_cli.sh kernel-config
  bash deploy/v2/kaggle_cli.sh dataset-version [message]
  bash deploy/v2/kaggle_cli.sh kernel-push
  bash deploy/v2/kaggle_cli.sh kernel-status
  bash deploy/v2/kaggle_cli.sh kernel-watch [timeout_sec] [interval_sec]
  bash deploy/v2/kaggle_cli.sh kernel-output
  bash deploy/v2/kaggle_cli.sh feedback [train_log_path]
  bash deploy/v2/kaggle_cli.sh full-cycle [message]

Key environment variables:
  KAGGLE_BIN, KAGGLE_CONFIG_DIR, KAGGLE_DATASET_PATH, KAGGLE_DATASET_ID,
  KAGGLE_KERNEL_PATH, KAGGLE_KERNEL_ID, KAGGLE_OUTPUT_DIR,
  KAGGLE_COMPETITION_SOURCE, KAGGLE_ENABLE_INTERNET, KAGGLE_ENABLE_GPU,
  KAGGLE_MACHINE_SHAPE, KAGGLE_KERNEL_TITLE, KAGGLE_NOTEBOOK_TEMPLATE
EOF
}

main() {
    local cmd="${1:-}"
    shift || true
    case "$cmd" in
        doctor) cmd_doctor "$@" ;;
        kernel-config) cmd_kernel_config "$@" ;;
        dataset-version) cmd_dataset_version "$@" ;;
        kernel-push) cmd_kernel_push "$@" ;;
        kernel-status) cmd_kernel_status "$@" ;;
        kernel-watch) cmd_kernel_watch "$@" ;;
        kernel-output) cmd_kernel_output "$@" ;;
        feedback) cmd_feedback "$@" ;;
        full-cycle) cmd_full_cycle "$@" ;;
        *) usage; [ -n "$cmd" ] && exit 1 ;;
    esac
}

main "$@"
