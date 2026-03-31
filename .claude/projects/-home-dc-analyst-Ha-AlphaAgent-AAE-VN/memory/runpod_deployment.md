---
name: Runpod AAE-VN Deployment
description: Current Runpod deployment details for AlphaAgentEvo training
type: project
---

Pod: bfvmbegnpaucub-64412112@ssh.runpod.io (SSH key: ~/Ha/.ssh/id_ed25519)
4× H100 80GB, SSH doesn't support command execution (web terminal only)
Repo: /workspace/AAE-VN (cloned from github.com/hongha5192-bit/AAE-VN, public)

**Why:** Reproducing AlphaAgentEvo paper for Vietnam stock market.

**How to apply:** All pod interaction must be via user pasting commands in web terminal. No SCP/SFTP/SSH command execution.

Training config: MODEL=Qwen/Qwen3-4B, TRAIN_BATCH_SIZE=20, PPO_MINI_BATCH_SIZE=20, TOTAL_STEPS=150, GPU_MEMORY_UTILIZATION=0.30, ROLLOUT_MAX_SEQS=4

Periods: train=2016-2020, val=2021, test=2022-2025
