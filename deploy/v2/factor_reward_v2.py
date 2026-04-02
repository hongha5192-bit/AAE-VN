"""Reward function that parses IR from tool_response tags in the model output.

Workaround for the Verl reward_scores pipeline bug where tool rewards
(computed by FactorTool) don't reach the reward manager. Instead, we
parse IR values directly from <tool_response> text in solution_str.

Reward components:
  - Format (0–0.3): valid <tool_call> with factor_name + factor_expr
  - Quality (0–0.7): based on best IR from <tool_response> tags
  - Penalties: malformed JSON, pseudo-calls, fabricated metrics, <think> tags
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ─── regex patterns ───
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
IR_RE = re.compile(r"IR=([-+]?\d*\.?\d+)")
IC_RE = re.compile(r"IC=([-+]?\d*\.?\d+)")
SUCCESS_RE = re.compile(r"^success:", re.IGNORECASE)
FAILED_RE = re.compile(r"^failed:", re.IGNORECASE)
PSEUDO_CALL_RE = re.compile(
    r"\b(?:evaluate_factor|evaluate factor)\s*\(", re.IGNORECASE
)
FABRICATED_RESULT_RE = re.compile(
    r"\b(?:best result|best factor|proposed factor|baseline|IR\s*[:=]|IC\s*[:=]|ICIR\s*[:=])",
    re.IGNORECASE,
)

MAX_TOOL_CALLS_PER_TURN = 4
PRETTY_LOGS = os.getenv("ALPHAEVO_PRETTY_REWARD_LOGS", "1") != "0"


def _preview(text: str, n: int = 180) -> str:
    compact = " ".join(str(text or "").split())
    return compact[:n] + "..." if len(compact) > n else compact


# ─── tool call validation (unchanged from original) ───

def _load_json_maybe(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(raw.replace("'", '"'))


def _validate_tool_payload(raw: str) -> str | None:
    if not raw:
        return "empty_payload"
    try:
        payload = _load_json_maybe(raw)
    except Exception:
        return "malformed_json"
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, dict):
        return "invalid_payload"
    name = payload.get("name") or (payload.get("function") or {}).get("name")
    if not name:
        return "missing_name"
    args = payload.get("arguments")
    if args is None:
        args = (payload.get("function") or {}).get("arguments")
    if isinstance(args, str):
        try:
            args = _load_json_maybe(args)
        except Exception:
            return "malformed_arguments"
    if not isinstance(args, dict):
        return "missing_arguments"
    if name != "evaluate_factor":
        return "wrong_tool"
    if not str(args.get("factor_expr", "")).strip():
        return "missing_factor_expr"
    if not str(args.get("factor_name", "")).strip():
        return "missing_factor_name"
    return None


def _count_valid_tool_calls(response: str) -> tuple[int, int, str | None]:
    """Returns (valid_count, total_count, first_error)."""
    matches = TOOL_CALL_RE.findall(response)
    if not matches:
        return 0, 0, None
    if len(matches) > MAX_TOOL_CALLS_PER_TURN:
        return 0, len(matches), "too_many_tool_calls"
    valid = 0
    first_err = None
    for m in matches:
        err = _validate_tool_payload(m.strip())
        if err is None:
            valid += 1
        elif first_err is None:
            first_err = err
    return valid, len(matches), first_err


# ─── tool response parsing (NEW) ───

def _parse_tool_responses(response: str) -> list[dict]:
    """Extract IR/IC values from <tool_response> tags."""
    results = []
    for match in TOOL_RESPONSE_RE.finditer(response):
        text = match.group(1).strip()
        entry = {"success": False, "ir": None, "ic": None}
        if SUCCESS_RE.match(text):
            entry["success"] = True
            ir_match = IR_RE.search(text)
            ic_match = IC_RE.search(text)
            if ir_match:
                try:
                    entry["ir"] = float(ir_match.group(1))
                except ValueError:
                    pass
            if ic_match:
                try:
                    entry["ic"] = float(ic_match.group(1))
                except ValueError:
                    pass
        elif FAILED_RE.match(text):
            entry["success"] = False
        results.append(entry)
    return results


def _quality_reward(tool_responses: list[dict], seed_ir: float) -> float:
    """Compute quality reward (0–0.7) based on best IR from tool responses.

    - No successful responses: 0.0
    - All negative IR, worse than seed: 0.05
    - Positive IR or better than seed: 0.1–0.7 (scaled by improvement)
    """
    successful = [r for r in tool_responses if r["success"] and r["ir"] is not None]
    if not successful:
        return 0.0

    best_ir = max(r["ir"] for r in successful)

    # Base quality: any successful backtest
    reward = 0.05

    # Improvement over seed (softplus scaling)
    improvement = best_ir - seed_ir
    # softplus: log(1 + exp(x)), capped
    perf = 0.1 * math.log1p(math.exp(min(improvement, 10.0)))
    reward += min(perf, 0.35)

    # Bonus for positive IR (absolute quality)
    if best_ir > 0:
        reward += min(0.15, best_ir * 0.15)

    # Bonus for multiple successful diverse evaluations
    if len(successful) >= 2:
        reward += 0.05
    if len(successful) >= 3:
        reward += 0.05

    return min(reward, 0.7)


# ─── main reward function ───

def compute_score(
    solution_str,
    ground_truth=None,
    method="flexible",
    format_score=0.0,
    score=1.0,
    extra_info=None,
    **kwargs,
):
    """Compute reward by parsing IR from tool_response tags in the response.

    Returns scalar in [-0.6, 1.0] range.
    """
    response = str(solution_str or "").strip()
    if not response:
        if PRETTY_LOGS:
            logger.warning("[reward-v2] empty_response reward=-0.60")
        return -0.6

    # Extract seed IR from extra_info if available
    seed_ir = 0.0
    if isinstance(extra_info, dict):
        seed_ir = float(extra_info.get("init_metric", 0.0) or 0.0)

    has_think = "<think>" in response.lower()

    # ── Stage 1: Format reward (0–0.3) ──
    valid_calls, total_calls, call_error = _count_valid_tool_calls(response)

    if valid_calls == 0:
        # No valid tool calls — penalize based on what went wrong
        if total_calls > MAX_TOOL_CALLS_PER_TURN:
            reward = -0.50
            reason = "too_many_tool_calls"
        elif call_error:
            reward = -0.50
            reason = call_error
        elif PSEUDO_CALL_RE.search(response):
            reward = -0.40
            reason = "pseudo_call"
        elif FABRICATED_RESULT_RE.search(response):
            reward = -0.45
            reason = "fabricated_metrics"
        else:
            reward = -0.35
            reason = "no_tool_call"
        if PRETTY_LOGS:
            logger.warning(
                "[reward-v2] %s reward=%.4f output=%s",
                reason, reward, _preview(response),
            )
        return reward

    # Valid tool calls exist
    format_reward = min(0.1 + 0.1 * valid_calls, 0.3)

    # Small penalty for <think> tags (model should output tool calls directly)
    if has_think:
        format_reward = max(format_reward - 0.1, 0.05)

    # ── Stage 2: Quality reward from tool_response (0–0.7) ──
    tool_responses = _parse_tool_responses(response)
    quality = _quality_reward(tool_responses, seed_ir)

    # ── Total ──
    total = format_reward + quality

    if PRETTY_LOGS:
        successful = [r for r in tool_responses if r["success"] and r.get("ir") is not None]
        best_ir = max((r["ir"] for r in successful), default=None)
        logger.warning(
            "[reward-v2] valid_calls=%d responses=%d best_ir=%s seed_ir=%.4f "
            "format=%.3f quality=%.3f total=%.4f",
            valid_calls, len(tool_responses), best_ir, seed_ir,
            format_reward, quality, total,
        )

    return min(total, 1.0)
