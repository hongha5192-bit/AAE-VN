#!/usr/bin/env python3
"""Launch deploy/v2/train.sh from a notebook-friendly Python process.

This avoids relying on %%bash / %%script output behavior in Jupyter frontends.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _read_export_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("export "):
            continue
        try:
            key, value = line[len("export ") :].split("=", 1)
        except ValueError:
            continue
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _find_repo() -> Path:
    env_repo = os.environ.get("REPO")
    if env_repo:
        p = Path(env_repo)
        if (p / "deploy/v2/train.sh").is_file():
            return p

    env_file_repo = _read_export_file(Path("/kaggle/working/aae_env.sh")).get("REPO")
    if env_file_repo:
        p = Path(env_file_repo)
        if (p / "deploy/v2/train.sh").is_file():
            return p

    candidates = [
        Path("/kaggle/input/datasets/gplebih/aae-new"),
        Path("/kaggle/input/datasets/giaphlm/aae-new"),
        Path("/kaggle/input/aae-new"),
    ]
    for p in candidates:
        if (p / "deploy/v2/train.sh").is_file():
            return p

    raise FileNotFoundError("Could not locate repo containing deploy/v2/train.sh")


def main() -> int:
    repo = _find_repo()
    env = os.environ.copy()
    env.update(_read_export_file(Path("/kaggle/working/aae_env.sh")))
    env.setdefault("WORK", "/kaggle/working/aae_v2")
    env.setdefault("TRAIN_STDOUT_MODE", "pretty")
    env.setdefault("TRAIN_HEARTBEAT_SEC", "10")
    env.setdefault("TRAIN_PRETTY_LOG", f"{env['WORK']}/logs/train.pretty.log")
    env.setdefault("ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE", "False")
    env.setdefault("MODEL", "Qwen/Qwen3-0.6B-MLX-bf16")
    env.setdefault("EXPERIMENT_NAME", "Qwen3-0.6B-MLX-bf16-verl-h100-paperish-50step")
    env.setdefault("TOTAL_STEPS", "50")
    env.setdefault("SAVE_FREQ", "50")
    env.setdefault("TEST_FREQ", "5")
    env.setdefault("VAL_BEFORE_TRAIN", "True")
    env.setdefault("TRAIN_BATCH_SIZE", "8")
    env.setdefault("VAL_BATCH_SIZE", "8")
    env.setdefault("PPO_MINI_BATCH_SIZE", "8")
    env.setdefault("ROLLOUT_N", "3")
    env.setdefault("VAL_ROLLOUT_N", "2")
    env.setdefault("MAX_PROMPT_LENGTH", "3072")
    env.setdefault("MAX_RESPONSE_LENGTH", "512")
    env.setdefault("MAX_MODEL_LEN", "4096")
    env.setdefault("ROLLOUT_MAX_BATCHED_TOKENS", "8192")
    env.setdefault("ROLLOUT_MAX_SEQS", "8")
    env.setdefault("PPO_MAX_TOKEN_LEN_PER_GPU", "2048")
    env.setdefault("FORWARD_MAX_TOKEN_LEN_PER_GPU", "2048")
    env.setdefault("LOG_PROB_MAX_TOKEN_LEN_PER_GPU", "2048")
    env.setdefault("ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU", "4")
    env.setdefault("ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU", "4")
    env.setdefault("REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU", "4")
    env.setdefault("CRITIC_FORWARD_MICRO_BATCH_SIZE_PER_GPU", "4")
    env.setdefault("CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU", "4")
    env.setdefault("GPU_MEMORY_UTILIZATION", "0.70")

    train_sh = repo / "deploy/v2/train.sh"
    cmd = ["/bin/bash", "-lc", f'exec bash "{train_sh}"']

    print(f"[notebook-live] repo={repo}", flush=True)
    print(f"[notebook-live] work={env['WORK']}", flush=True)
    print(
        f"[notebook-live] stdout_mode={env.get('TRAIN_STDOUT_MODE')} "
        f"heartbeat_sec={env.get('TRAIN_HEARTBEAT_SEC')} "
        f"pretty_log={env.get('TRAIN_PRETTY_LOG')} "
        f"inference_chat={env.get('ROLLOUT_USE_INFERENCE_CHAT_TEMPLATE')}",
        flush=True,
    )
    print(
        f"[notebook-live] preset steps={env.get('TOTAL_STEPS')} test_freq={env.get('TEST_FREQ')} "
        f"batch={env.get('TRAIN_BATCH_SIZE')} rollout_n={env.get('ROLLOUT_N')} "
        f"val_rollout_n={env.get('VAL_ROLLOUT_N')} gpu_mem_util={env.get('GPU_MEMORY_UTILIZATION')}",
        flush=True,
    )
    print(f"[notebook-live] cmd={' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        print("[notebook-live] interrupt received, terminating child process", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise

    rc = proc.wait()
    print(f"[notebook-live] exit_code={rc}", flush=True)
    return rc


if __name__ == "__main__":
    rc = main()
    if rc != 0:
        raise SystemExit(rc)
