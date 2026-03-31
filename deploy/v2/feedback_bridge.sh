#!/bin/bash
# Bridge Kaggle outputs -> local autofeedback -> optional dataset push
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KAGGLE_CLI="$SCRIPT_DIR/kaggle_cli.sh"

KAGGLE_OUTPUT_DIR="${KAGGLE_OUTPUT_DIR:-$REPO_ROOT/.kaggle-output}"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/deploy/v2/feedback}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
POLL_SEC="${POLL_SEC:-30}"
UPLOAD_FEEDBACK="${UPLOAD_FEEDBACK:-0}"

mkdir -p "$KAGGLE_OUTPUT_DIR" "$REPORT_DIR"

echo "==> Watching kernel until terminal status (timeout=${TIMEOUT_SEC}s, poll=${POLL_SEC}s)"
set +e
bash "$KAGGLE_CLI" kernel-watch "$TIMEOUT_SEC" "$POLL_SEC"
WATCH_RC=$?
set -e

STATUS_LINE="$(bash "$KAGGLE_CLI" kernel-status 2>&1 || true)"
echo "Kernel status: $STATUS_LINE"

echo "==> Downloading latest kernel outputs"
set +e
bash "$KAGGLE_CLI" kernel-output
OUT_RC=$?
set -e
if [ $OUT_RC -ne 0 ]; then
    echo "WARN: kernel-output failed (rc=$OUT_RC)"
fi

find_train_log() {
    local root="$1"
    local candidates=(
        "$root/train.log"
        "$root/logs/train.log"
        "$root/aae_v2/logs/train.log"
    )
    local c
    for c in "${candidates[@]}"; do
        if [ -f "$c" ]; then
            echo "$c"
            return 0
        fi
    done
    find "$root" -type f -name 'train.log' 2>/dev/null | head -n1
}

TRAIN_LOG="$(find_train_log "$KAGGLE_OUTPUT_DIR" || true)"
TS="$(date '+%Y%m%d_%H%M%S')"
REPORT_TXT="$REPORT_DIR/autofeedback_${TS}.txt"
REPORT_MD="$REPORT_DIR/autofeedback_${TS}.md"
LATEST_MD="$REPORT_DIR/latest_autofeedback.md"

{
    echo "# Kaggle Autofeedback"
    echo
    echo "- Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- Kernel status: \`$STATUS_LINE\`"
    echo "- Watch return code: \`$WATCH_RC\`"
    echo "- Output dir: \`$KAGGLE_OUTPUT_DIR\`"
    if [ -n "${TRAIN_LOG:-}" ] && [ -f "$TRAIN_LOG" ]; then
        echo "- Train log: \`$TRAIN_LOG\`"
    else
        echo "- Train log: **not found**"
    fi
    echo
    echo "## Autofeedback"
    echo
} > "$REPORT_MD"

if [ -n "${TRAIN_LOG:-}" ] && [ -f "$TRAIN_LOG" ]; then
    python3 "$SCRIPT_DIR/autofeedback.py" --train-log "$TRAIN_LOG" --tail-lines 1200 | tee "$REPORT_TXT"
    {
        echo '```text'
        cat "$REPORT_TXT"
        echo '```'
    } >> "$REPORT_MD"
else
    echo "No train.log found under $KAGGLE_OUTPUT_DIR" | tee "$REPORT_TXT"
    {
        echo '```text'
        cat "$REPORT_TXT"
        echo '```'
    } >> "$REPORT_MD"
fi

cp "$REPORT_MD" "$LATEST_MD"
echo "Report written:"
echo "  $REPORT_MD"
echo "  $LATEST_MD"

if [ "$UPLOAD_FEEDBACK" = "1" ]; then
    echo "==> Uploading feedback report back to Kaggle dataset"
    bash "$KAGGLE_CLI" dataset-version "autofeedback sync: $TS"
fi
