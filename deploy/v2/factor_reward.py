"""Fallback reward function for responses that never reached the real tool path.

When the multi-turn tool execution succeeds, `patch_verl.py` makes Verl prefer the
trajectory reward returned by `FactorTool.execute()`.

This fallback is only for malformed generations, so it should be opinionated:
- valid `evaluate_factor` tool call: small positive reward
- malformed tool call JSON: negative reward
- prose / pseudo-calls / fabricated metrics: clear negative reward
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
PSEUDO_CALL_RE = re.compile(r"\b(?:evaluate_factor|evaluate factor|Evaluate_factor|Evaluate factor)\s*\(", re.IGNORECASE)
FABRICATED_RESULT_RE = re.compile(
    r"\b(?:best result|best factor|proposed factor|baseline|IR\s*[:=]|IC\s*[:=]|ICIR\s*[:=])",
    re.IGNORECASE,
)
MAX_TOOL_CALLS_PER_TURN = 4
PRETTY_INVALID_LOGS = os.getenv("ALPHAEVO_PRETTY_INVALID_LOGS", "1") != "0"
PRETTY_INVALID_EXCERPT_CHARS = int(os.getenv("ALPHAEVO_PRETTY_INVALID_EXCERPT_CHARS", "180"))


def _preview(text: str) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= PRETTY_INVALID_EXCERPT_CHARS:
        return compact
    return compact[: PRETTY_INVALID_EXCERPT_CHARS - 3] + "..."


def _pretty_log_invalid(kind: str, reward: float, reason: str | None, response: str) -> None:
    if not PRETTY_INVALID_LOGS:
        return
    logger.warning(
        "[response-live] status=%s reward=%.4f reason=%s output=%s",
        kind,
        reward,
        reason or "none",
        _preview(response),
    )


def _load_json_maybe(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = raw.replace("'", '"')
        return json.loads(fixed)


def _validate_tool_payload(raw_payload: str) -> str | None:
    if not raw_payload:
        return "empty_payload"

    try:
        payload = _load_json_maybe(raw_payload)
    except Exception:
        logger.debug("Malformed tool call JSON: %s", raw_payload[:200])
        return "malformed_json"

    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, dict):
        return "invalid_payload"

    name = payload.get("name") or payload.get("tool_name")
    function = payload.get("function")
    if not name and isinstance(function, dict):
        name = function.get("name")
    if not name:
        return "missing_name"

    arguments = payload.get("arguments")
    if arguments is None and isinstance(function, dict):
        arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = _load_json_maybe(arguments)
        except Exception:
            return "malformed_arguments"
    if not isinstance(arguments, dict):
        return "missing_arguments"

    factor_name = str(arguments.get("factor_name", "")).strip()
    factor_expr = str(arguments.get("factor_expr", "")).strip()
    if name != "evaluate_factor":
        return "wrong_tool"
    if not factor_expr:
        return "missing_factor_expr"
    if not factor_name:
        return "missing_factor_name"
    return None


def _extract_tagged_tool_payloads(response: str) -> tuple[list[str], str | None]:
    matches = TOOL_CALL_RE.findall(response)
    payloads: list[str] = [m.strip() for m in matches if m and m.strip()]
    if not payloads:
        return [], None
    if len(payloads) > MAX_TOOL_CALLS_PER_TURN:
        return [], "too_many_tool_calls"

    return payloads, None


def _extract_raw_json_payloads(response: str) -> list[str]:
    payloads: list[str] = []
    decoder = json.JSONDecoder()
    i = 0
    while i < len(response):
        if response[i] != "{":
            i += 1
            continue
        try:
            obj, consumed = decoder.raw_decode(response[i:])
        except Exception:
            i += 1
            continue
        try:
            payloads.append(json.dumps(obj, ensure_ascii=False))
        except Exception:
            pass
        i += max(consumed, 1)
    return payloads


def _count_valid_payloads(payloads: list[str]) -> tuple[int, str | None]:
    if not payloads:
        return 0, None

    valid_count = 0
    first_error = None
    for raw_payload in payloads:
        error = _validate_tool_payload(raw_payload)
        if error is None:
            valid_count += 1
        elif first_error is None:
            first_error = error

    if valid_count > 0:
        return valid_count, None
    return 0, first_error


def _format_adherence_score(response: str) -> float:
    score = 0.0
    stripped = response.lstrip()
    lower = response.lower()

    open_count = len(re.findall(r"<tool_call>", lower))
    close_count = len(re.findall(r"</tool_call>", lower))
    if stripped.startswith("<tool_call>"):
        score += 0.35
    if open_count > 0:
        score += 0.20
    if open_count > 0 and open_count == close_count:
        score += 0.15
    if "evaluate_factor" in lower:
        score += 0.15
    if "factor_name" in lower:
        score += 0.10
    if "factor_expr" in lower:
        score += 0.10
    if "<think>" in lower or "</think>" in lower:
        score -= 0.30
    if FABRICATED_RESULT_RE.search(response):
        score -= 0.15

    # Small dense shaping to avoid reward plateaus when outputs are all invalid.
    score += min(len(stripped), 512) / 5120.0
    return max(0.0, min(score, 1.0))


def compute_score(solution_str, ground_truth=None, method="flexible", format_score=0.0, score=1.0, **kwargs):
    """Return a scalar fallback reward in roughly PPO-friendly range [-1, 1]."""
    response = str(solution_str or "").strip()
    if not response:
        reward = -0.6
        _pretty_log_invalid("empty_response", reward, "empty_response", response)
        return reward

    adherence = _format_adherence_score(response)
    has_think = "<think>" in response.lower() or "</think>" in response.lower()
    tagged_payloads, tag_error_reason = _extract_tagged_tool_payloads(response)
    tagged_valid_count, tagged_payload_error = _count_valid_payloads(tagged_payloads)

    if tagged_valid_count > 0 and response.lstrip().startswith("<tool_call>") and not has_think:
        return min(0.3 + 0.1 * (tagged_valid_count - 1), 0.6)

    if tagged_valid_count > 0:
        reward = -0.15 + 0.20 * adherence
        reason = "thinking_trace_present" if has_think else "tool_call_not_first_token"
        _pretty_log_invalid("invalid_tool_call", reward, reason, response)
        return reward

    if tag_error_reason is not None:
        reward = -0.58 + 0.20 * adherence
        _pretty_log_invalid("invalid_tool_call", reward, tag_error_reason, response)
        return reward

    if tagged_payload_error is not None:
        reward = -0.58 + 0.20 * adherence
        _pretty_log_invalid("invalid_tool_call", reward, tagged_payload_error, response)
        return reward

    raw_json_valid_count, raw_json_error = _count_valid_payloads(_extract_raw_json_payloads(response))
    if raw_json_valid_count > 0:
        reward = -0.30 + 0.15 * adherence
        _pretty_log_invalid("untagged_json_call", reward, "missing_tool_call_tags", response)
        return reward

    if raw_json_error is not None:
        reward = -0.58 + 0.20 * adherence
        _pretty_log_invalid("invalid_tool_call", reward, raw_json_error, response)
        return reward

    if PSEUDO_CALL_RE.search(response):
        reward = -0.45 + 0.20 * adherence
        _pretty_log_invalid("pseudo_call", reward, "pseudo_call", response)
        return reward

    if FABRICATED_RESULT_RE.search(response):
        reward = -0.45
        _pretty_log_invalid("fabricated_metrics", reward, "fabricated_metrics", response)
        return reward

    reward = -0.35 + 0.30 * adherence
    _pretty_log_invalid("freeform_response", reward, "freeform_response", response)
    return reward
