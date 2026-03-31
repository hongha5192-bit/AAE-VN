#!/usr/bin/env python3
"""Kaggle kernel entrypoint for AlphaAgentEvo v2."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def sh(cmd: str, *, log_file: Path | None = None) -> None:
    print(f"+ {cmd}", flush=True)
    wrapped_cmd = f"set -o pipefail; PYTHONUNBUFFERED=1 stdbuf -oL -eL {cmd}"
    proc = subprocess.Popen(
        ["/bin/bash", "-lc", wrapped_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_handle = None
    try:
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_file.open("a", encoding="utf-8", buffering=1)
            print(f"[live-log] {log_file}", flush=True)

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log_handle is not None:
                log_handle.write(line)

        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    finally:
        if log_handle is not None:
            log_handle.close()


def find_repo_input() -> Path:
    candidates = [
        Path("/kaggle/input/datasets/gplebih/aae-new"),
        Path("/kaggle/input/datasets/gplebih/aae-new/AAE-new"),
        Path("/kaggle/input/datasets/giaphlm/aae-new"),
        Path("/kaggle/input/datasets/giaphlm/aae-new/AAE-new"),
        Path("/kaggle/input/aae-new"),
        Path("/kaggle/input/aae-new/AAE-new"),
    ]
    for path in candidates:
        if path.exists() and (path / "deploy/v2/setup.sh").is_file():
            return path

    for setup_sh in Path("/kaggle/input").rglob("setup.sh"):
        setup_sh = setup_sh.resolve()
        if str(setup_sh).endswith("/deploy/v2/setup.sh"):
            return setup_sh.parents[2]

    for p in Path("/kaggle/input").rglob("AAE-new"):
        if p.is_dir() and (p / "deploy/v2/setup.sh").is_file():
            return p
    raise FileNotFoundError("Could not find AAE-new repo under /kaggle/input")


def main() -> None:
    src_repo = find_repo_input()
    dst_repo = Path("/kaggle/working/AAE-new")
    logs_dir = Path("/kaggle/working/aae_v2/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    if dst_repo.exists():
        shutil.rmtree(dst_repo)
    shutil.copytree(src_repo, dst_repo, symlinks=False)

    os.chdir(dst_repo)
    sh("bash deploy/v2/setup.sh", log_file=logs_dir / "setup.live.log")
    sh("bash deploy/v2/train.sh", log_file=logs_dir / "train.live.log")
    sh("bash deploy/v2/infer.sh", log_file=logs_dir / "infer.live.log")
    sh(
        "python3 deploy/v2/autofeedback.py --train-log /kaggle/working/aae_v2/logs/train.live.log",
        log_file=logs_dir / "autofeedback.live.log",
    )


if __name__ == "__main__":
    main()
