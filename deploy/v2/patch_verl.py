#!/usr/bin/env python3
"""Patch Verl v0.4.1 and pinned sglang for the Kaggle AlphaAgentEvo flow."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


FACTOR_METRICS_BLOCK = """


def compute_factor_validation_metrics(
    data_sources: list[str],
    sample_inputs: list[str],
    infos_dict: dict[str, list[Any]],
    ks: tuple[int, ...] = (3, 5),
) -> dict[str, dict[str, float]]:
    \"\"\"Compute factor-specific validation metrics such as VR and pass@k.\"\"\"
    required_keys = {\"valid_proposal\", \"best_trial_metric\", \"seed_metric\"}
    if not required_keys.issubset(infos_dict):
        return {}

    grouped = defaultdict(list)
    for idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[idx]
        grouped[(data_source, prompt)].append(
            {
                \"valid_proposal\": float(infos_dict[\"valid_proposal\"][idx]),
                \"best_trial_metric\": float(infos_dict[\"best_trial_metric\"][idx]),
                \"seed_metric\": float(infos_dict[\"seed_metric\"][idx]),
            }
        )

    aggregated = defaultdict(lambda: defaultdict(list))
    for (data_source, _prompt), items in grouped.items():
        proposal_count = len(items)
        valid_scores = [item[\"best_trial_metric\"] for item in items if item[\"valid_proposal\"] > 0.5]
        seed_metric = items[0][\"seed_metric\"] if items else 0.0

        valid_ratio = (len(valid_scores) / proposal_count) if proposal_count > 0 else 0.0
        beat_rate = float(any(score > seed_metric for score in valid_scores))
        aggregated[data_source][\"vr\"].append(valid_ratio)
        aggregated[data_source][\"valid_count_mean\"].append(float(len(valid_scores)))
        aggregated[data_source][\"proposal_count_mean\"].append(float(proposal_count))
        aggregated[data_source][\"best_metric_mean\"].append(float(max(valid_scores)) if valid_scores else 0.0)
        aggregated[data_source][\"beat_rate\"].append(beat_rate)
        aggregated[data_source][\"improved_any_mean\"].append(beat_rate)

        valid_scores.sort(reverse=True)
        for k in ks:
            if proposal_count < k:
                continue
            top_scores = valid_scores[:k]
            aggregated[data_source][f\"pass@{k}\"].append(float(any(score > seed_metric for score in top_scores)))

    return {
        data_source: {metric_name: float(np.mean(values)) for metric_name, values in metrics.items()}
        for data_source, metrics in aggregated.items()
    }
"""


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Could not find expected block in {path}")
    path.write_text(text.replace(old, new))


def patch_naive_reward_manager(verl_dir: Path) -> None:
    path = verl_dir / "verl/workers/reward_manager/naive.py"
    old = """            ground_truth = data_item.non_tensor_batch[\"reward_model\"][\"ground_truth\"]\n            data_source = data_item.non_tensor_batch[self.reward_fn_key]\n            extra_info = data_item.non_tensor_batch.get(\"extra_info\", None)\n\n            score = self.compute_score(\n                data_source=data_source,\n                solution_str=response_str,\n                ground_truth=ground_truth,\n                extra_info=extra_info,\n            )\n"""
    new = """            ground_truth = data_item.non_tensor_batch[\"reward_model\"][\"ground_truth\"]\n            data_source = data_item.non_tensor_batch[self.reward_fn_key]\n            extra_info = data_item.non_tensor_batch.get(\"extra_info\", None)\n\n            tool_reward = None\n            if \"reward_scores\" in data_item.non_tensor_batch:\n                tool_results = data_item.non_tensor_batch[\"reward_scores\"]\n                if hasattr(tool_results, \"item\") and not isinstance(tool_results, dict):\n                    try:\n                        tool_results = tool_results.item()\n                    except Exception:\n                        pass\n                if isinstance(tool_results, dict):\n                    _raw_reward = tool_results.get(\"evaluate_factor\")\n                    if isinstance(_raw_reward, dict):\n                        tool_reward = _raw_reward.get(\"score\")\n                    elif isinstance(_raw_reward, (int, float)):\n                        tool_reward = float(_raw_reward)\n\n            score = self.compute_score(\n                data_source=data_source,\n                solution_str=response_str,\n                ground_truth=ground_truth,\n                extra_info=extra_info,\n            )\n\n            if tool_reward is not None:\n                score = tool_reward\n"""
    replace_once(path, old, new)


def patch_metric_utils(verl_dir: Path) -> None:
    path = verl_dir / "verl/trainer/ppo/metric_utils.py"
    text = path.read_text()
    if "def compute_factor_validation_metrics(" in text:
        return
    marker = "\n    return data_src2var2metric2val\n"
    if marker not in text:
        raise RuntimeError(f"Could not find marker in {path}")
    path.write_text(text.replace(marker, marker + FACTOR_METRICS_BLOCK))


def patch_ray_trainer(verl_dir: Path) -> None:
    path = verl_dir / "verl/trainer/ppo/ray_trainer.py"
    text = path.read_text()

    import_old = """from verl.trainer.ppo.metric_utils import (\n    compute_data_metrics,\n    compute_throughout_metrics,\n    compute_timing_metrics,\n    process_validation_metrics,\n)\n"""
    import_new = """from verl.trainer.ppo.metric_utils import (\n    compute_data_metrics,\n    compute_factor_validation_metrics,\n    compute_throughout_metrics,\n    compute_timing_metrics,\n    process_validation_metrics,\n)\n"""
    if "compute_factor_validation_metrics" not in text:
        if import_old not in text:
            raise RuntimeError(f"Could not find metric_utils import block in {path}")
        text = text.replace(import_old, import_new)

    old = """        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)\n        metric_dict = {}\n        for data_source, var2metric2val in data_src2var2metric2val.items():\n            core_var = \"acc\" if \"acc\" in var2metric2val else \"reward\"\n            for var_name, metric2val in var2metric2val.items():\n                n_max = max([int(name.split(\"@\")[-1].split(\"/\")[0]) for name in metric2val.keys()])\n                for metric_name, metric_val in metric2val.items():\n                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in [\"mean\", \"maj\", \"best\"]) and (f\"@{n_max}\" in metric_name):\n                        metric_sec = \"val-core\"\n                    else:\n                        metric_sec = \"val-aux\"\n                    pfx = f\"{metric_sec}/{data_source}/{var_name}/{metric_name}\"\n                    metric_dict[pfx] = metric_val\n\n        return metric_dict\n"""
    new = """        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)\n        metric_dict = {}\n        for data_source, var2metric2val in data_src2var2metric2val.items():\n            core_var = \"acc\" if \"acc\" in var2metric2val else \"reward\"\n            for var_name, metric2val in var2metric2val.items():\n                n_max = max([int(name.split(\"@\")[-1].split(\"/\")[0]) for name in metric2val.keys()])\n                for metric_name, metric_val in metric2val.items():\n                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in [\"mean\", \"maj\", \"best\"]) and (f\"@{n_max}\" in metric_name):\n                        metric_sec = \"val-core\"\n                    else:\n                        metric_sec = \"val-aux\"\n                    pfx = f\"{metric_sec}/{data_source}/{var_name}/{metric_name}\"\n                    metric_dict[pfx] = metric_val\n\n        factor_eval_metrics = compute_factor_validation_metrics(\n            data_sources=data_sources,\n            sample_inputs=sample_inputs,\n            infos_dict=reward_extra_infos_dict,\n            ks=(3, 5),\n        )\n        for data_source, metric2val in factor_eval_metrics.items():\n            for metric_name, metric_val in metric2val.items():\n                metric_dict[f\"val-paper/{data_source}/{metric_name}\"] = metric_val\n\n        return metric_dict\n"""

    if "val-paper/" not in text:
        if old not in text:
            raise RuntimeError(f"Could not find validation metric block in {path}")
        text = text.replace(old, new)

    path.write_text(text)


def patch_fsdp_sglang(verl_dir: Path) -> None:
    path = verl_dir / "verl/workers/sharding_manager/fsdp_sglang.py"
    text = path.read_text()
    helper = """

def _get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
"""
    if "def _get_or_create_event_loop():" not in text:
        marker = "logger.setLevel(os.getenv(\"VERL_LOGGING_LEVEL\", \"WARN\"))\n"
        if marker not in text:
            raise RuntimeError(f"Could not find logger marker in {path}")
        text = text.replace(marker, marker + helper)

    # Repair previously bad patch that accidentally introduced self-recursion.
    text = text.replace("        return _get_or_create_event_loop()\n", "        return asyncio.get_event_loop()\n")

    # Only replace call sites, not helper internals.
    text = text.replace("loop = asyncio.get_event_loop()", "loop = _get_or_create_event_loop()")
    path.write_text(text)


def patch_torch_functional(verl_dir: Path) -> None:
    path = verl_dir / "verl/utils/torch_functional.py"
    text = path.read_text()
    if "labels = labels.long()" in text:
        return
    needle = "    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:\n"
    replacement = "    labels = labels.long()\n    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:\n"
    if needle not in text:
        raise RuntimeError(f"Could not find logprobs block in {path}")
    path.write_text(text.replace(needle, replacement, 1))


def patch_sglang_rollout_utils(verl_dir: Path) -> None:
    path = verl_dir / "verl/workers/rollout/sglang_rollout/utils.py"
    text = path.read_text()
    if "byte_arr = np.frombuffer(serialized_data, dtype=np.uint8).copy()" in text:
        return

    old_candidates = [
        "            tensor_data = torch.ByteTensor(np.frombuffer(serialized_data, dtype=np.uint8)).to(device)\n",
        "    tensor_data = torch.ByteTensor(np.frombuffer(serialized_data, dtype=np.uint8)).to(device)\n",
    ]
    for old in old_candidates:
        if old in text:
            indent = old.split("tensor_data", 1)[0]
            new = (
                f"{indent}# Make a writable copy to avoid undefined behavior when downstream writes tensor buffers.\n"
                f"{indent}byte_arr = np.frombuffer(serialized_data, dtype=np.uint8).copy()\n"
                f"{indent}tensor_data = torch.ByteTensor(byte_arr).to(device)\n"
            )
            path.write_text(text.replace(old, new, 1))
            return

    raise RuntimeError(f"Could not find broadcast tensor conversion block in {path}")


def patch_sglang_rollout_tool_kwargs(verl_dir: Path) -> None:
    path = verl_dir / "verl/workers/rollout/sglang_rollout/sglang_rollout.py"
    text = path.read_text()
    if '.get(tool_call.function.name, {}).get("execute_kwargs", {})' in text:
        return

    old = '                                **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {}),\n'
    new = '                                **(_req.tools_kwargs or {}).get(tool_call.function.name, {}).get("execute_kwargs", {}),\n'
    replace_once(path, old, new)


def patch_fsdp_checkpoint_manager(verl_dir: Path) -> None:
    path = verl_dir / "verl/utils/checkpoint/fsdp_checkpoint_manager.py"
    old = """            if unwrap_model.can_generate() and hasattr(model_config, \"name_or_path\") and model_config.name_or_path:\n                # Some model's name_or_path is empty if not initialized from pretrained,\n                # in this cases, we don't save generation config.\n                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)\n                generation_config.save_pretrained(hf_config_tokenizer_path)\n            else:\n                generation_config = None\n"""
    new = """            if unwrap_model.can_generate() and hasattr(model_config, \"name_or_path\") and model_config.name_or_path:\n                # Some model's name_or_path is empty if not initialized from pretrained,\n                # in this cases, we don't save generation config.\n                try:\n                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)\n                    generation_config.save_pretrained(hf_config_tokenizer_path)\n                except Exception as exc:\n                    generation_config = None\n                    log_with_rank(\n                        f\"Skipping generation config save for {model_config.name_or_path}: {exc}\",\n                        rank=self.rank,\n                        logger=logger,\n                        log_only_rank_0=True,\n                    )\n            else:\n                generation_config = None\n"""
    replace_once(path, old, new)


def patch_dp_actor_flash_attn(verl_dir: Path) -> None:
    path = verl_dir / "verl/workers/actor/dp_actor.py"
    text = path.read_text()
    if "_FLASH_ATTN_BERT_PADDING_AVAILABLE" in text:
        return

    old_import = """if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
"""
    new_import = """_FLASH_ATTN_BERT_PADDING_AVAILABLE = False

if is_cuda_available:
    try:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        _FLASH_ATTN_BERT_PADDING_AVAILABLE = True
    except Exception:
        index_first_axis = pad_input = rearrange = unpad_input = None
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
    _FLASH_ATTN_BERT_PADDING_AVAILABLE = True
"""
    if old_import not in text:
        raise RuntimeError(f"Could not find flash_attn import block in {path}")
    text = text.replace(old_import, new_import)

    old_cfg = """        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
"""
    new_cfg = """        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if self.use_remove_padding and not _FLASH_ATTN_BERT_PADDING_AVAILABLE:
            if torch.distributed.get_rank() == 0:
                logger.warning("flash_attn.bert_padding is unavailable; force use_remove_padding=False")
            self.use_remove_padding = False
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
"""
    if old_cfg not in text:
        raise RuntimeError(f"Could not find use_remove_padding block in {path}")
    text = text.replace(old_cfg, new_cfg)
    path.write_text(text)


def patch_dp_critic_flash_attn(verl_dir: Path) -> None:
    path = verl_dir / "verl/workers/critic/dp_critic.py"
    text = path.read_text()
    if "_FLASH_ATTN_BERT_PADDING_AVAILABLE" in text:
        return

    old_import = """if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
"""
    new_import = """_FLASH_ATTN_BERT_PADDING_AVAILABLE = False

if is_cuda_available:
    try:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        _FLASH_ATTN_BERT_PADDING_AVAILABLE = True
    except Exception:
        index_first_axis = pad_input = rearrange = unpad_input = None
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
    _FLASH_ATTN_BERT_PADDING_AVAILABLE = True
"""
    if old_import not in text:
        raise RuntimeError(f"Could not find flash_attn import block in {path}")
    text = text.replace(old_import, new_import)

    old_cfg = """        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")
"""
    new_cfg = """        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        if self.use_remove_padding and not _FLASH_ATTN_BERT_PADDING_AVAILABLE:
            logger.warning("flash_attn.bert_padding is unavailable; force use_remove_padding=False")
            self.use_remove_padding = False
        print(f"Critic use_remove_padding={self.use_remove_padding}")
"""
    if old_cfg not in text:
        raise RuntimeError(f"Could not find use_remove_padding block in {path}")
    text = text.replace(old_cfg, new_cfg)
    path.write_text(text)


def patch_sglang_quantization() -> None:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        print("WARNING: Could not locate installed sglang package; skipping quantization patch")
        return

    package_root = Path(spec.origin).resolve().parent
    path = package_root / "srt/layers/quantization/__init__.py"
    text = path.read_text()
    if "COMPRESSED_TENSORS_AVAILABLE" in text:
        return

    old_import = """from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (\n    CompressedTensorsConfig,\n)\n"""
    new_import = """try:\n    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (\n        CompressedTensorsConfig,\n    )\n    COMPRESSED_TENSORS_AVAILABLE = True\nexcept Exception:\n    CompressedTensorsConfig = None\n    COMPRESSED_TENSORS_AVAILABLE = False\n"""
    if old_import not in text:
        raise RuntimeError(f"Could not find compressed tensors import in {path}")
    text = text.replace(old_import, new_import)

    old_entry = '    "compressed-tensors": CompressedTensorsConfig,\n'
    if old_entry not in text:
        raise RuntimeError(f"Could not find compressed tensors registry entry in {path}")
    text = text.replace(old_entry, "")

    marker = "}\n\n# VLLM-dependent quantization methods\n"
    if marker not in text:
        raise RuntimeError(f"Could not find quantization methods marker in {path}")
    text = text.replace(
        marker,
        "}\nif COMPRESSED_TENSORS_AVAILABLE:\n    BASE_QUANTIZATION_METHODS[\"compressed-tensors\"] = CompressedTensorsConfig\n\n# VLLM-dependent quantization methods\n",
    )
    path.write_text(text)


def patch_fsdp_workers_checkpoint_safety(verl_dir: Path) -> None:
    path = verl_dir / "verl/workers/fsdp_workers.py"
    text = path.read_text()
    if "checkpoint save failed (non-fatal)" in text:
        return

    old = """        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        dist.barrier()"""
    new = """        try:
            self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        except Exception as exc:
            log_with_rank(f"checkpoint save failed (non-fatal): {exc}", rank=0, logger=logger)
        dist.barrier()"""
    if old not in text:
        return  # silently skip if pattern doesn't match
    path.write_text(text.replace(old, new, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-dir", required=True)
    args = parser.parse_args()

    verl_dir = Path(args.verl_dir).resolve()
    if not verl_dir.exists():
        raise FileNotFoundError(verl_dir)

    patch_naive_reward_manager(verl_dir)
    patch_metric_utils(verl_dir)
    patch_ray_trainer(verl_dir)
    patch_fsdp_sglang(verl_dir)
    patch_torch_functional(verl_dir)
    patch_sglang_rollout_utils(verl_dir)
    patch_sglang_rollout_tool_kwargs(verl_dir)
    patch_fsdp_checkpoint_manager(verl_dir)
    patch_fsdp_workers_checkpoint_safety(verl_dir)
    patch_dp_actor_flash_attn(verl_dir)
    patch_dp_critic_flash_attn(verl_dir)
    patch_sglang_quantization()
    print(f"Patched runtime at {verl_dir}")


if __name__ == "__main__":
    main()
