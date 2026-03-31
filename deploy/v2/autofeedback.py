#!/usr/bin/env python3
"""Auto-feedback for Kaggle training logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PATTERNS = [
    (
        "missing_module",
        re.compile(r"ModuleNotFoundError: No module named '([^']+)'"),
        "Missing Python module. Check pinned versions and runtime patches.",
    ),
    (
        "ray_task_error",
        re.compile(r"ray\.exceptions\.RayTaskError"),
        "Ray worker failed. Inspect worker stderr logs for the first real exception.",
    ),
    (
        "disk_full",
        re.compile(r"No space left on device|ENOSPC"),
        "Disk is full. Remove old checkpoints/logs before retry.",
    ),
    (
        "oom",
        re.compile(r"CUDA out of memory|OutOfMemoryError|signal: Killed"),
        "Likely OOM. Reduce rollout tokens/batch size or memory utilization.",
    ),
    (
        "token_budget_mismatch",
        re.compile(r"max_model_len should be greater than total sequence length|Invalid token budget", re.IGNORECASE),
        "Token budget mismatch. Ensure MAX_MODEL_LEN >= MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH.",
    ),
    (
        "empty_train_dataloader",
        re.compile(r"Train dataloader is empty", re.IGNORECASE),
        "All prompts were filtered out. Increase MAX_PROMPT_LENGTH or inspect dataset prompt length.",
    ),
    (
        "sys_excepthook_loop",
        re.compile(r"Error in sys\.excepthook"),
        "Exception reporting is broken in worker. Read /tmp/ray/session_latest/logs/worker-*.err for root cause.",
    ),
    (
        "train_done",
        re.compile(r"Training Progress:\s*100%|global_step_20"),
        "Training appears to have completed.",
    ),
]


def _tail(path: Path, lines: int) -> str:
    text = path.read_text(errors="replace")
    if lines <= 0:
        return text
    return "\n".join(text.splitlines()[-lines:])


def analyze(log_text: str) -> list[tuple[str, str, str]]:
    findings: list[tuple[str, str, str]] = []
    for key, regex, advice in PATTERNS:
        match = regex.search(log_text)
        if match:
            excerpt = match.group(0)
            findings.append((key, excerpt, advice))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", default="/kaggle/working/aae_v2/logs/train.log")
    parser.add_argument("--tail-lines", type=int, default=500)
    args = parser.parse_args()

    log_path = Path(args.train_log)
    if not log_path.is_file():
        print(f"ERROR: train log not found: {log_path}")
        return 1

    text = _tail(log_path, args.tail_lines)
    findings = analyze(text)
    if not findings:
        print("No known error signatures found in the inspected log segment.")
        return 0

    print("Auto-feedback findings:")
    for idx, (key, excerpt, advice) in enumerate(findings, start=1):
        print(f"{idx}. [{key}] {excerpt}")
        print(f"   Advice: {advice}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
