"""FactorTool for AlphaAgentEvo v2 on official Verl v0.4.1."""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import requests

try:
    from verl.utils.rollout_trace import rollout_trace_op
except Exception:
    # Verl v0.4.1 does not expose rollout_trace; keep tool callable without tracing.
    def rollout_trace_op(func):
        return func

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Reward caps and coefficients (aligned with AlphaAgentEvo Eq.5 / Appendix D)
CAP_TOOL = 1.0
CAP_CONS = 0.2
CAP_EXPL = 0.3
CAP_PERF = 0.5
CAP_STREAK = 0.6

ALPHA_SUCC = 0.1
ALPHA_FAIL = 0.2
ALPHA_CONS = 0.02
ALPHA_EXP = 0.02
ALPHA_PERF = 0.1
ALPHA_STREAK = 0.15

H_LOW = 0.1
H_HIGH = 0.9
R_TOOL_FLOOR = 0.01

try:
    from expression_manager.factor_ast import (  # type: ignore
        count_all_nodes,
        find_largest_common_subtree,
        parse_expression as parse_ast,
    )
    _AST_SIM_AVAILABLE = True
except Exception:
    _AST_SIM_AVAILABLE = False
    from difflib import SequenceMatcher


class FactorTool(BaseTool):
    """Backtest alpha factors and return trajectory-level rewards."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.instance_retention_sec = float(
            os.getenv(
                "ALPHAEVO_TOOL_INSTANCE_TTL_SEC",
                config.get("instance_retention_sec", 1800),
            )
        )
        self.backtest_api_url = os.getenv(
            "ALPHAEVO_BACKTEST_API_URL",
            config.get("backtest_api_url", "http://localhost:8002/backtest"),
        )
        self.backtest_defaults = {
            "backtest_start_time": config.get("backtest_start_time", "2016-01-01"),
            "backtest_end_time": config.get("backtest_end_time", "2023-12-31"),
            "stock_pool": config.get("stock_pool", "VN100"),
            "start_cash": float(config.get("start_cash", 10_000_000.0)),
        }
        self.request_timeout_sec = float(
            os.getenv(
                "ALPHAEVO_BACKTEST_TIMEOUT_SEC",
                config.get("request_timeout_sec", 120),
            )
        )
        self.pretty_logs = os.getenv("ALPHAEVO_PRETTY_TOOL_LOGS", "1") != "0"
        self.pretty_expr_chars = int(os.getenv("ALPHAEVO_PRETTY_EXPR_CHARS", "180"))
        self.max_tool_calls_per_turn = int(
            os.getenv(
                "MAX_TOOL_CALLS_PER_TURN",
                config.get("max_tool_calls_per_turn", 4),
            )
        )
        logger.info("Initializing FactorTool with name: %s", self.tool_schema.function.name)
        logger.info("Tool config: %s", config)

    @staticmethod
    def _instance_label(instance_id: str) -> str:
        return str(instance_id).split("-", 1)[0]

    def _expr_preview(self, expr: Any) -> str:
        text = " ".join(str(expr or "").split())
        if len(text) <= self.pretty_expr_chars:
            return text
        return text[: self.pretty_expr_chars - 3] + "..."

    def _pretty_log(self, message: str, *args: Any) -> None:
        if self.pretty_logs:
            logger.warning(message, *args)

    def _new_instance_state(self, **kwargs) -> dict[str, Any]:
        init_metric = self._safe_float(kwargs.get("init_metric", 0.0))
        return {
            "factor_name": "",
            "factor_expr": "",
            "metric": "Information_Ratio_with_cost",
            "init_metric": init_metric,
            "best_metric": init_metric,
            "best_trial_metric": float("-inf"),
            "best_trial_result": None,
            "init_factor_expr": kwargs.get("init_factor_expr", ""),
            "reward": {},
            "streak": 0,
            "tool_call_count": 0,
            "succ_tried_factors": [],
            "failed_count": 0,
            "_released": False,
            "_created_at": time.time(),
        }

    def _prune_released_instances(self) -> None:
        if self.instance_retention_sec <= 0:
            return
        now = time.time()
        expired_ids = [
            instance_id
            for instance_id, instance in self._instance_dict.items()
            if instance.get("_released")
            and now - self._safe_float(instance.get("released_at", 0.0), 0.0) > self.instance_retention_sec
        ]
        for instance_id in expired_ids:
            del self._instance_dict[instance_id]

    def _ensure_instance(self, instance_id: str, **kwargs) -> dict[str, Any]:
        self._prune_released_instances()
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            instance = self._new_instance_state(**kwargs)
            instance["_recreated"] = True
            self._instance_dict[instance_id] = instance
            logger.warning(
                "[FactorTool] missing instance_id=%s; recreated tool state to avoid rollout crash",
                instance_id,
            )
            return instance

        if instance.get("_released"):
            instance["_released"] = False
            instance.pop("released_at", None)
            logger.warning(
                "[FactorTool] revive released instance_id=%s; execute arrived after release",
                instance_id,
            )
        return instance

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._prune_released_instances()
        self._instance_dict[instance_id] = self._new_instance_state(**kwargs)
        logger.warning("[FactorTool] create instance_id=%s", instance_id)
        instance = self._instance_dict[instance_id]
        self._pretty_log(
            "[factor-live] create id=%s seed_metric=%.4f seed_expr=%s",
            self._instance_label(instance_id),
            self._safe_float(instance.get("init_metric", 0.0)),
            self._expr_preview(instance.get("init_factor_expr", "")),
        )
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        factor_name = parameters.get("factor_name", "")
        factor_expr = parameters.get("factor_expr", "")
        metric = parameters.get("metric", "Information_Ratio_with_cost")

        instance = self._ensure_instance(instance_id, **kwargs)
        instance["factor_name"] = factor_name
        instance["factor_expr"] = factor_expr
        instance["metric"] = metric
        instance["tool_call_count"] += 1
        call_count = int(instance["tool_call_count"])

        self._pretty_log(
            "[factor-live] eval id=%s call=%d/%d factor=%s expr=%s",
            self._instance_label(instance_id),
            call_count,
            self.max_tool_calls_per_turn,
            factor_name,
            self._expr_preview(factor_expr),
        )

        metric_value, status, metrics = await self._call_backtest_api(instance_id)

        if status == "success":
            result = {
                "factor_expr": factor_expr,
                "metric_value": metric_value,
                "metrics": metrics,
            }
            instance["succ_tried_factors"].append(result)
            if metric_value > instance["best_trial_metric"]:
                instance["best_trial_metric"] = metric_value
                instance["best_trial_result"] = result
            if metric_value > instance["best_metric"]:
                instance["best_metric"] = metric_value
                instance["streak"] += 1

            reward_summary = await self.calc_reward(instance_id)
            step_reward = float(reward_summary["score"])
            self._pretty_log(
                "[factor-live] result id=%s call=%d/%d status=success reward=%.4f ir=%.4f ic=%.4f icir=%.4f best=%.4f seed=%.4f improved=%d",
                self._instance_label(instance_id),
                call_count,
                self.max_tool_calls_per_turn,
                step_reward,
                metrics.get("ir", 0.0),
                metrics.get("ic", 0.0),
                metrics.get("icir", 0.0),
                self._safe_float(reward_summary.get("best_trial_metric", 0.0)),
                self._safe_float(reward_summary.get("seed_metric", 0.0)),
                int(self._safe_float(reward_summary.get("improved_over_seed", 0.0)) > 0.5),
            )
            response_text = (
                f'success: Evaluated factor "{factor_name}" with expression "{factor_expr}", '
                f'IR={metrics.get("ir", 0.0):.4f}, IC={metrics.get("ic", 0.0):.4f}, '
                f'ICIR={metrics.get("icir", 0.0):.4f}'
            )
            return response_text, step_reward, reward_summary

        instance["failed_count"] += 1
        reward_summary = await self.calc_reward(instance_id)
        self._pretty_log(
            "[factor-live] result id=%s call=%d/%d status=failed reward=%.4f reason=%s expr=%s",
            self._instance_label(instance_id),
            call_count,
            self.max_tool_calls_per_turn,
            float(reward_summary["score"]),
            status,
            self._expr_preview(factor_expr),
        )
        return (
            f"failed: factor {factor_name} with expression {factor_expr}. Reason: {status}",
            float(reward_summary["score"]),
            reward_summary,
        )

    async def _call_backtest_api(self, instance_id: str) -> tuple[float, str, dict[str, float]]:
        instance = self._ensure_instance(instance_id)
        test_request = {
            "exprs": {instance["factor_name"]: instance["factor_expr"]},
            "backtest_start_time": self.backtest_defaults["backtest_start_time"],
            "backtest_end_time": self.backtest_defaults["backtest_end_time"],
            "stock_pool": self.backtest_defaults["stock_pool"],
            "start_cash": self.backtest_defaults["start_cash"],
        }

        t0 = time.time()
        logger.warning(
            "[FactorTool] backtest start name=%s expr_len=%d timeout=%.1fs",
            instance["factor_name"],
            len(instance["factor_expr"]),
            self.request_timeout_sec,
        )
        try:
            response = requests.post(
                self.backtest_api_url,
                json=test_request,
                timeout=self.request_timeout_sec,
            )
            if response.status_code != 200:
                logger.warning(
                    "[FactorTool] backtest http error name=%s status=%s elapsed=%.2fs",
                    instance["factor_name"],
                    response.status_code,
                    time.time() - t0,
                )
                return 0.0, f"http_{response.status_code}", self._normalize_metrics({})

            result = response.json()
            if not result.get("data"):
                detail = result.get("detail", {})
                logger.warning(
                    "[FactorTool] backtest empty data name=%s reason=%s elapsed=%.2fs",
                    instance["factor_name"],
                    str(detail.get("error", "empty_data"))[:120],
                    time.time() - t0,
                )
                return 0.0, str(detail.get("error", "empty_data"))[:200], self._normalize_metrics({})

            metrics = self._normalize_metrics(result["data"].get("metrics", {}))
            metric_name = instance["metric"]
            metric_value = self._safe_float(result["data"].get("metrics", {}).get(metric_name, 0.0))
            logger.warning(
                "[FactorTool] backtest success name=%s metric=%s value=%.4f elapsed=%.2fs",
                instance["factor_name"],
                metric_name,
                metric_value,
                time.time() - t0,
            )
            return float(round(metric_value, 4)), "success", metrics
        except Exception as exc:
            logger.error(
                "Backtest exception name=%s elapsed=%.2fs err=%s",
                instance["factor_name"],
                time.time() - t0,
                exc,
            )
            return 0.0, str(exc)[:200], self._normalize_metrics({})

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            casted = float(value)
            if np.isnan(casted) or np.isinf(casted):
                return default
            return casted
        except (TypeError, ValueError):
            return default

    def _normalize_metrics(self, metrics: dict[str, Any] | None) -> dict[str, float]:
        metrics = metrics or {}
        return {
            "ir": self._safe_float(metrics.get("Information_Ratio_with_cost", 0.0)),
            "ic": self._safe_float(metrics.get("IC", 0.0)),
            "icir": self._safe_float(metrics.get("ICIR", 0.0)),
            "annualized_return": self._safe_float(metrics.get("Annualized_Return_with_cost", 0.0)),
            "mdd": self._safe_float(metrics.get("Max_Drawdown_with_cost", 0.0)),
        }

    @staticmethod
    def _expr_similarity(expr_a: str, expr_b: str) -> float:
        if not expr_a or not expr_b:
            return 0.0
        if _AST_SIM_AVAILABLE:
            try:
                ast_a = parse_ast(expr_a)
                ast_b = parse_ast(expr_b)
                match = find_largest_common_subtree(ast_a, ast_b)
                size_a = count_all_nodes(expr_a)
                size_b = count_all_nodes(expr_b)
                if match is None or max(size_a, size_b) == 0:
                    return 0.0
                return float(match.size / max(size_a, size_b))
            except Exception:
                return 0.0
        return float(SequenceMatcher(None, expr_a, expr_b).ratio())

    @staticmethod
    def _similarity(expr: str, candidates: list[str]) -> float | None:
        if not expr or not candidates:
            return None
        sims = [FactorTool._expr_similarity(expr, candidate) for candidate in candidates if candidate]
        if not sims:
            return None
        return max(sims)

    async def calc_reward(self, instance_id: str, **kwargs) -> dict[str, float]:
        instance = self._ensure_instance(instance_id, **kwargs)
        init_metric = self._safe_float(instance.get("init_metric", 0.0), 0.0)
        successful = list(instance.get("succ_tried_factors", []))
        success_count = len(successful)
        fail_count = int(instance.get("failed_count", 0))
        call_count = int(instance.get("tool_call_count", 0))
        fail_count = max(fail_count, call_count - success_count)

        best_trial_metric = instance["best_trial_metric"] if success_count > 0 else init_metric
        best_trial_result = instance.get("best_trial_result")
        best_trial_metrics = best_trial_result.get("metrics", {}) if isinstance(best_trial_result, dict) else {}
        # Paper Eq.(5): R = (R_cons + R_expl) / R_tool + R_perf * R_streak (all capped)
        r_tool_raw = ALPHA_SUCC * success_count - ALPHA_FAIL * fail_count
        r_tool_capped = min(r_tool_raw, CAP_TOOL)
        r_tool_denom = max(r_tool_capped, R_TOOL_FLOOR)

        seed_expr = str(instance.get("init_factor_expr", "") or "")
        successful_exprs = [str(item.get("factor_expr", "") or "") for item in successful]

        r_cons = 0.0
        for expr in successful_exprs:
            if not expr:
                continue
            sim = self._expr_similarity(expr, seed_expr)
            if H_LOW < sim < H_HIGH:
                r_cons += ALPHA_CONS

        r_expl = 0.0
        prior_exprs = [seed_expr]
        for expr in successful_exprs:
            if not expr:
                prior_exprs.append(expr)
                continue
            max_sim = max((self._expr_similarity(expr, p) for p in prior_exprs if p), default=0.0)
            r_expl += ALPHA_EXP * (1.0 - max_sim)
            prior_exprs.append(expr)

        baseline = init_metric
        x = max(-20.0, min(20.0, best_trial_metric - baseline))
        r_perf = ALPHA_PERF * math.log1p(math.exp(x))
        r_streak = ALPHA_STREAK * float(instance.get("streak", 0))

        r_cons_capped = min(r_cons, CAP_CONS)
        r_expl_capped = min(r_expl, CAP_EXPL)
        r_perf_capped = min(r_perf, CAP_PERF)
        r_streak_capped = min(r_streak, CAP_STREAK)

        direction_term = (r_cons_capped + r_expl_capped) / r_tool_denom
        quality_term = r_perf_capped * r_streak_capped
        total_reward = float(direction_term + quality_term)
        total_reward = float(np.clip(total_reward, -1.0, 8.0))

        reward_summary = {
            "score": total_reward,
            "ir": self._safe_float(best_trial_metrics.get("ir", 0.0)),
            "ic": self._safe_float(best_trial_metrics.get("ic", 0.0)),
            "icir": self._safe_float(best_trial_metrics.get("icir", 0.0)),
            "best_trial_metric": self._safe_float(best_trial_metric, 0.0),
            "seed_metric": self._safe_float(init_metric, 0.0),
            "valid_proposal": float(success_count > 0),
            "improved_over_seed": float(success_count > 0 and best_trial_metric > init_metric),
            "success_count": float(success_count),
            "tool_call_count": float(call_count),
            "failed_count": float(fail_count),
            "r_tool": float(r_tool_raw),
            "r_cons": float(r_cons_capped),
            "r_expl": float(r_expl_capped),
            "r_perf": float(r_perf_capped),
            "r_streak": float(r_streak_capped),
            "direction_term": float(direction_term),
            "quality_term": float(quality_term),
        }
        instance["reward"] = reward_summary

        logger.info(
            "Reward for %s: %s (r_tool=%.4f, denom=%.4f, r_cons=%.4f, r_expl=%.4f, "
            "r_perf=%.4f, r_streak=%.4f, direction=%.4f, quality=%.4f)",
            instance_id,
            reward_summary,
            r_tool_raw,
            r_tool_denom,
            r_cons_capped,
            r_expl_capped,
            r_perf_capped,
            r_streak_capped,
            direction_term,
            quality_term,
        )
        return reward_summary

    async def release(self, instance_id: str, **kwargs) -> None:
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            logger.warning("[FactorTool] release missing instance_id=%s", instance_id)
            return
        instance["_released"] = True
        instance["released_at"] = time.time()
        logger.warning(
            "[FactorTool] release instance_id=%s retained_for=%.1fs",
            instance_id,
            self.instance_retention_sec,
        )
        self._prune_released_instances()
