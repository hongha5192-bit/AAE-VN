#!/bin/bash
set -euo pipefail

WORK="${WORK:-/kaggle/working/aae_v2}"
LOGS_DIR="${LOGS_DIR:-$WORK/logs}"
RAY_LOG_DIR="${RAY_LOG_DIR:-/tmp/ray/session_latest/logs}"

echo "=================== TRAIN ERROR CHECK ==================="
echo "WORK=$WORK"
echo "LOGS_DIR=$LOGS_DIR"
echo "RAY_LOG_DIR=$RAY_LOG_DIR"

if [ -f "$LOGS_DIR/train.live.log" ]; then
    echo "--- tail train.live.log ---"
    tail -n 120 "$LOGS_DIR/train.live.log" || true
fi

if [ -f "$LOGS_DIR/api.log" ]; then
    echo "--- tail api.log ---"
    tail -n 80 "$LOGS_DIR/api.log" || true
fi

python - <<'PY'
from pathlib import Path
import os

ray_log_dir = Path(os.environ.get("RAY_LOG_DIR", "/tmp/ray/session_latest/logs"))
if not ray_log_dir.exists():
    print(f"[error-check] ray log dir not found: {ray_log_dir}")
    raise SystemExit(0)

err_files = sorted(
    ray_log_dir.glob("worker-*.err"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)[:8]

if not err_files:
    print("[error-check] no worker-*.err files found")
    raise SystemExit(0)

printed = 0
for path in err_files:
    lines = path.read_text(errors="replace").splitlines()
    trace_idxs = [i for i, line in enumerate(lines) if line.startswith("Traceback (most recent call last):")]
    if not trace_idxs:
        continue

    selected = None
    for idx in trace_idxs:
        prev_idx = idx - 1
        while prev_idx >= 0 and not lines[prev_idx].strip():
            prev_idx -= 1
        prev = lines[prev_idx] if prev_idx >= 0 else ""
        if "During handling of the above exception" not in prev and "Original exception was" not in prev:
            selected = idx
            break
    if selected is None:
        selected = trace_idxs[0]

    start = max(0, selected - 6)
    end = min(len(lines), selected + 120)
    print(f"\n=== first traceback candidate: {path} ===")
    for line in lines[start:end]:
        print(line)
    printed += 1

if printed == 0:
    print("[error-check] no traceback blocks found in recent worker stderr files")
PY

echo "========================================================="
