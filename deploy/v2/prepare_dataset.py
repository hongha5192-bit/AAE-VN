#!/usr/bin/env python3
"""Normalize AlphaAgentEvo parquet files for Verl v0.4.1 multi-turn tools."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


TOOL_NAME = "evaluate_factor"
PERIOD_CONFIGS = {
    "train": {"start": "2016-01-01", "end": "2023-12-31"},
    "val": {"start": "2024-01-01", "end": "2024-12-31"},
    "test": {"start": "2025-01-01", "end": "2026-12-31"},
}
FACTOR_RE = re.compile(r"^Factor:\s*(.+)$", re.MULTILINE)
EXPR_RE = re.compile(r"^Expression:\s*(.+)$", re.MULTILINE)
BASELINE_RE = re.compile(r"^Baseline IR:\s*([+-]?\d+(?:\.\d+)?)$", re.MULTILINE)


class SeedMetricEvaluator:
    def __init__(self, repo_root: Path, cache_path: Path):
        self.repo_root = repo_root
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self._stats: dict[str, dict[str, int]] = {}
        self._execute_expression = None

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        records = payload.get("records", {})
        return records if isinstance(records, dict) else {}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "records": self.cache,
        }
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _make_key(self, split_name: str, seed_expr: str) -> str:
        digest = hashlib.sha1(seed_expr.encode("utf-8")).hexdigest()
        return f"{split_name}:{digest}"

    def _split_stats(self, split_name: str) -> dict[str, int]:
        return self._stats.setdefault(
            split_name,
            {"cached": 0, "evaluated": 0, "failed": 0},
        )

    def _ensure_executor(self) -> None:
        if self._execute_expression is not None:
            return

        repo_root_str = str(self.repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from backtest.factor_executor import configure_periods, load_data, execute_expression  # type: ignore

        configure_periods(PERIOD_CONFIGS)
        load_data()
        self._execute_expression = execute_expression

    def evaluate(self, *, split_name: str, seed_name: str, seed_expr: str, fallback_metric: float) -> float:
        key = self._make_key(split_name, seed_expr)
        stats = self._split_stats(split_name)
        cached = self.cache.get(key)
        if isinstance(cached, dict) and "seed_metric" in cached:
            stats["cached"] += 1
            return self._safe_float(cached.get("seed_metric"), fallback_metric)

        self._ensure_executor()
        with contextlib.redirect_stdout(io.StringIO()):
            result = self._execute_expression(seed_expr, period=split_name)
        if result.get("success"):
            metric = self._safe_float(result.get("ir"), fallback_metric)
            self.cache[key] = {
                "split": split_name,
                "seed_name": seed_name,
                "seed_metric": metric,
                "error": None,
            }
            stats["evaluated"] += 1
            return metric

        stats["failed"] += 1
        self.cache[key] = {
            "split": split_name,
            "seed_name": seed_name,
            "seed_metric": fallback_metric,
            "error": str(result.get("error", "unknown"))[:300],
        }
        print(
            f"[seed-ir] WARN split={split_name} factor={seed_name} fallback_metric={fallback_metric:.4f} "
            f"reason={str(result.get('error', 'unknown'))[:160]}",
            flush=True,
        )
        return fallback_metric

    def print_summary(self, split_name: str, values: pd.Series, *, source: str = "backtest", precomputed: int = 0) -> None:
        stats = self._split_stats(split_name)
        series = pd.to_numeric(values, errors="coerce")
        print(
            f"[seed-ir] split={split_name} source={source} precomputed={precomputed} "
            f"cached={stats['cached']} evaluated={stats['evaluated']} failed={stats['failed']} "
            f"mean={series.mean():.4f} min={series.min():.4f} max={series.max():.4f}",
            flush=True,
        )


def load_system_prompt(repo_root: Path) -> str:
    prompt_path = repo_root / "training" / "system_prompt.md"
    return prompt_path.read_text().strip()


def _as_prompt_list(raw_prompt: Any) -> list[dict[str, Any]]:
    if hasattr(raw_prompt, "tolist"):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, str):
        try:
            raw_prompt = json.loads(raw_prompt)
        except json.JSONDecodeError:
            return []
    if isinstance(raw_prompt, list):
        return [item for item in raw_prompt if isinstance(item, dict)]
    return []


def _first_message(prompt: Any, role: str) -> str:
    for message in _as_prompt_list(prompt):
        if message.get("role") == role:
            return str(message.get("content", "")).strip()
    return ""


def _extract_seed_name(row: pd.Series) -> str:
    if "seed_name" in row and pd.notna(row["seed_name"]):
        return str(row["seed_name"]).strip()

    user_msg = _first_message(row.get("prompt"), "user")
    match = FACTOR_RE.search(user_msg)
    if match:
        return match.group(1).strip()
    return "seed_factor"


def _extract_seed_expr(row: pd.Series) -> str:
    raw_tools_kwargs = row.get("tools_kwargs")
    if isinstance(raw_tools_kwargs, dict):
        tool_cfg = raw_tools_kwargs.get(TOOL_NAME, raw_tools_kwargs)
        if isinstance(tool_cfg, dict):
            create_kwargs = tool_cfg.get("create_kwargs", tool_cfg)
            if isinstance(create_kwargs, dict):
                expr = str(create_kwargs.get("init_factor_expr", "")).strip()
                if expr:
                    return expr

    if "seed_expr" in row and pd.notna(row["seed_expr"]):
        return str(row["seed_expr"]).strip()

    user_msg = _first_message(row.get("prompt"), "user")
    match = EXPR_RE.search(user_msg)
    if match:
        return match.group(1).strip()
    return ""


def _extract_seed_metric(row: pd.Series) -> float:
    raw_tools_kwargs = row.get("tools_kwargs")
    if isinstance(raw_tools_kwargs, dict):
        tool_cfg = raw_tools_kwargs.get(TOOL_NAME, raw_tools_kwargs)
        if isinstance(tool_cfg, dict):
            create_kwargs = tool_cfg.get("create_kwargs", tool_cfg)
            if isinstance(create_kwargs, dict):
                try:
                    return float(create_kwargs.get("init_metric", 0.0))
                except (TypeError, ValueError):
                    pass

    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        try:
            return float(reward_model.get("ground_truth", 0.0))
        except (TypeError, ValueError):
            pass

    if "seed_ir" in row and pd.notna(row["seed_ir"]):
        return float(row["seed_ir"])

    user_msg = _first_message(row.get("prompt"), "user")
    match = BASELINE_RE.search(user_msg)
    if match:
        return float(match.group(1))
    return 0.0


def normalize_tools_kwargs(raw_tools_kwargs: object, *, seed_expr: str, seed_metric: float) -> dict:
    if isinstance(raw_tools_kwargs, dict):
        tool_cfg = raw_tools_kwargs.get(TOOL_NAME, raw_tools_kwargs)
        if isinstance(tool_cfg, dict):
            create_kwargs = tool_cfg.get("create_kwargs", tool_cfg)
            if isinstance(create_kwargs, dict):
                return {
                    TOOL_NAME: {
                        "create_kwargs": {
                            "init_factor_expr": str(create_kwargs.get("init_factor_expr", seed_expr)),
                            # Always align the tool baseline with the normalized seed metric.
                            # Some shipped parquet files still carry stale init_metric=0.0.
                            "init_metric": float(seed_metric),
                        }
                    }
                }

    return {
        TOOL_NAME: {
            "create_kwargs": {
                "init_factor_expr": seed_expr,
                "init_metric": float(seed_metric),
            }
        }
    }


def build_user_prompt(seed_name: str, seed_expr: str, seed_metric: float) -> str:
    example_payload = json.dumps(
        {
            "name": TOOL_NAME,
            "arguments": {
                "factor_name": "seed_baseline",
                "factor_expr": seed_expr,
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"Evolve the following seed alpha factor to beat its baseline IR.\n\n"
        f"Factor: {seed_name}\n"
        f"Expression: {seed_expr}\n"
        f"Baseline IR: {seed_metric:.4f}\n\n"
        f"Rules for your next assistant message:\n"
        f"- It must be a valid call to `evaluate_factor`.\n"
        f"- Output only the tool call. No prose, no markdown, no analysis.\n"
        f"- Output from 1 to 4 `<tool_call>...</tool_call>` blocks.\n"
        f"- The first non-whitespace token must be `<tool_call>`.\n"
        f"- Do not output `<think>` or any reasoning text.\n"
        f"- Raw JSON without surrounding `<tool_call>...</tool_call>` tags is invalid and will not execute.\n"
        f"- Inside `<tool_call>`, use JSON: "
        f"{{\"name\":\"evaluate_factor\",\"arguments\":{{\"factor_name\":\"...\",\"factor_expr\":\"...\"}}}}.\n"
        f"- Do not fabricate IR, IC, ICIR, or a best-result summary.\n"
        f"- If you need a baseline, call the seed expression first.\n"
        f"- If you choose a variation, keep it syntactically valid and close to the seed.\n\n"
        f"Canonical example:\n"
        f"<tool_call>{example_payload}</tool_call>"
    )


def normalize_prompt(row: pd.Series, system_prompt: str) -> list[dict[str, str]]:
    seed_name = _extract_seed_name(row)
    seed_expr = _extract_seed_expr(row)
    if "seed_ir" in row and pd.notna(row["seed_ir"]):
        seed_metric = float(row["seed_ir"])
    else:
        seed_metric = _extract_seed_metric(row)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(seed_name, seed_expr, seed_metric)},
    ]


def normalize_parquet(
    src_path: Path,
    dst_path: Path,
    repo_root: Path,
    *,
    seed_metric_evaluator: SeedMetricEvaluator | None = None,
    force_recompute_seed_ir: bool = False,
) -> None:
    system_prompt = load_system_prompt(repo_root)
    df = pd.read_parquet(src_path).copy()
    split_name = src_path.stem

    if "prompt" not in df.columns:
        raise ValueError(f"{src_path} does not contain prompt")

    df["seed_name"] = df.apply(_extract_seed_name, axis=1)
    df["seed_expr"] = df.apply(_extract_seed_expr, axis=1)
    df["seed_ir_raw"] = df.apply(_extract_seed_metric, axis=1)
    precomputed_seed_ir = pd.to_numeric(df["seed_ir"], errors="coerce") if "seed_ir" in df.columns else pd.Series(pd.NA, index=df.index, dtype="float64")

    if seed_metric_evaluator is not None and split_name in PERIOD_CONFIGS:
        precomputed_count = int(precomputed_seed_ir.notna().sum())
        total_rows = len(df)
        if not force_recompute_seed_ir and precomputed_count == total_rows:
            df["seed_ir"] = precomputed_seed_ir.astype(float)
            seed_metric_evaluator.print_summary(
                split_name,
                df["seed_ir"],
                source="input_parquet",
                precomputed=precomputed_count,
            )
        else:
            if precomputed_count and not force_recompute_seed_ir:
                print(
                    f"[seed-ir] split={split_name} using precomputed values for "
                    f"{precomputed_count}/{total_rows} rows; evaluating missing rows only",
                    flush=True,
                )

            seed_metrics: list[float] = []
            for idx, row in enumerate(df.itertuples(index=False), start=1):
                precomputed_metric = precomputed_seed_ir.iloc[idx - 1]
                if not force_recompute_seed_ir and pd.notna(precomputed_metric):
                    metric = float(precomputed_metric)
                else:
                    metric = seed_metric_evaluator.evaluate(
                        split_name=split_name,
                        seed_name=str(getattr(row, "seed_name")),
                        seed_expr=str(getattr(row, "seed_expr")),
                        fallback_metric=SeedMetricEvaluator._safe_float(getattr(row, "seed_ir_raw")),
                    )
                seed_metrics.append(metric)
                if idx == 1 or idx == total_rows or idx % 25 == 0:
                    print(
                        f"[seed-ir] progress split={split_name} row={idx}/{total_rows} "
                        f"factor={str(getattr(row, 'seed_name'))} ir={metric:.4f}",
                        flush=True,
                    )
            df["seed_ir"] = seed_metrics
            seed_metric_evaluator.print_summary(
                split_name,
                df["seed_ir"],
                source="mixed" if precomputed_count and not force_recompute_seed_ir else "backtest",
                precomputed=precomputed_count if not force_recompute_seed_ir else 0,
            )
    else:
        df["seed_ir"] = df["seed_ir_raw"]

    df["prompt"] = df.apply(lambda row: normalize_prompt(row, system_prompt), axis=1)

    if "tools_kwargs" in df.columns:
        df["tools_kwargs"] = df.apply(
            lambda row: normalize_tools_kwargs(
                row["tools_kwargs"],
                seed_expr=str(row["seed_expr"]),
                seed_metric=float(row["seed_ir"]),
            ),
            axis=1,
        )

    if "reward_model" in df.columns:
        df["reward_model"] = df.apply(
            lambda row: {"ground_truth": f"{float(row['seed_ir']):.6f}"},
            axis=1,
        )

    if "seed_ir_raw" in df.columns:
        df = df.drop(columns=["seed_ir_raw"])

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", required=False)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--force-recompute-seed-ir", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else input_dir.parent.parent.resolve()
    cache_path = output_dir / "seed_metric_cache.json"
    seed_metric_evaluator = SeedMetricEvaluator(repo_root=repo_root, cache_path=cache_path)
    splits = [split.strip() for split in str(args.splits).split(",") if split.strip()]

    for split in splits:
        src_path = input_dir / f"{split}.parquet"
        dst_path = output_dir / f"{split}.parquet"
        normalize_parquet(
            src_path,
            dst_path,
            repo_root,
            seed_metric_evaluator=seed_metric_evaluator,
            force_recompute_seed_ir=args.force_recompute_seed_ir,
        )
        print(f"Normalized {src_path} -> {dst_path}")

    seed_metric_evaluator.save()
    print(f"[seed-ir] cache_saved={cache_path}")


if __name__ == "__main__":
    main()
