"""
Evolve Alpha158 seed factors using the trained Qwen3-4B model.

Flow:
  For each Alpha158 seed:
    1. Backtest seed on VN data → get init IC
    2. Feed seed + IC to Qwen → model evolves it via tool calls
    3. Run up to N_TURNS evolution turns
    4. Save all evolved factors and best result

Usage:
    python evolve_alpha158.py [--seeds N] [--turns N] [--metric IC]
"""

import json
import os
import time
import argparse
import numpy as np
import requests
from datetime import datetime
from openai import OpenAI
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_API    = "http://localhost:8100/v1"
BACKTEST_API = "http://localhost:8002/backtest"  # single-factor proxy (legacy)
COMBO_API    = "http://localhost:8003/combo_backtest"  # multi-factor LGBModel combo
MODEL_NAME   = "qwen3-alpha"

SEEDS_FILE  = Path(__file__).parent / "run22_seeds.jsonl"
RESULTS_DIR = Path(__file__).parent / "evo_results"
RESULTS_DIR.mkdir(exist_ok=True)

# VN data range
BT_START = "2022-01-01"
BT_END   = "2025-12-31"

# "Keep 2, Evolve 1" — fixed factors that stay constant during evolution
FIXED_FACTORS = {}

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "evaluate_factor",
        "description": "Evaluate a factor expression by backtesting on Vietnam stock market data. Returns the IC (Information Coefficient) metric.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor_name": {"type": "string", "description": "Descriptive name for the factor"},
                "factor_expr": {"type": "string", "description": "Factor expression using $open,$close,$high,$low,$volume,$return,$net_foreign_val,$net_foreign_vol"},
            },
            "required": ["factor_name", "factor_expr"],
        },
    },
}

# Variables available in VN data
VN_VARS = "$open, $close, $high, $low, $volume, $return, $net_foreign_val, $net_foreign_vol"

SYSTEM_PROMPT = f"""You are a quantitative researcher specializing in Vietnam stock market alpha factors.
You are evolving one factor from a 3-factor ensemble. These 3 factors are combined as features in a LightGBM model to predict stock returns. Your goal is to improve the TARGET factor while complementing the other two.

## Available Variables
{VN_VARS}

## Available Functions
Time-series: TS_MEAN, TS_STD, TS_MAX, TS_MIN, TS_SUM, TS_RANK, TS_CORR, TS_QUANTILE, TS_ARGMAX, TS_ARGMIN, TS_ZSCORE, TS_PCTCHANGE, DECAYLINEAR, EMA
Cross-sectional: RANK, ZSCORE
Math: DELAY, ABS, LOG, EXP, SQRT, SIGN, MAX, MIN
Statistics: MEDIAN, PERCENTILE, REGBETA

## Rules
- ALWAYS use $ prefix for variables (e.g. $close, NOT close)
- Only use variables listed above — no $amount, $vwap, etc.
- Each turn: propose 2-3 evolved variants and call evaluate_factor for each
- Analyze backtest feedback and improve systematically
- Avoid high correlation with the other 2 ensemble factors — diversity improves ensemble performance
- Target: improve the proxy metric AND maintain low correlation with the other factors"""


client = OpenAI(api_key="EMPTY", base_url=MODEL_API, timeout=300)


def backtest(name: str, expr: str, metric: str = "IC") -> tuple[float, str]:
    """Run backtest via single-factor proxy API."""
    req = {
        "exprs": {name: expr},
        "backtest_start_time": BT_START,
        "backtest_end_time": BT_END,
        "stock_pool": "VN100",
        "update_freq": 5,
        "label_forward_days": 5,
        "use_cache": True,
    }
    try:
        r = requests.post(BACKTEST_API, json=req, timeout=120)
        if r.status_code == 200:
            data = r.json().get("data", {})
            if data and data.get("metrics"):
                val = data["metrics"].get(metric, np.nan)
                return float(np.round(val, 4)), "success"
            return np.nan, "no metrics"
        return np.nan, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return np.nan, str(e)


def combo_backtest(name: str, expr: str, metric: str = "IR_without_cost") -> tuple[float, str]:
    """Run combo backtest: fixed factors + this evolved factor → LGBModel → IR.

    Uses the combo API (port 8003) which trains LGBModel on all factors together
    and returns portfolio metrics. This is the "Keep 2, Evolve 1" approach.
    """
    # Build factor_exprs: fixed factors + the new evolved one
    factor_exprs = dict(FIXED_FACTORS)
    factor_exprs[name] = expr

    req = {"factor_exprs": factor_exprs}
    try:
        r = requests.post(COMBO_API, json=req, timeout=300)
        if r.status_code == 200:
            data = r.json().get("data", {})
            if data and data.get("success"):
                val = data.get(metric, np.nan)
                return float(np.round(val, 4)), "success"
            return np.nan, data.get("error", "no metrics")
        return np.nan, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return np.nan, str(e)


def execute_tool_calls(tool_calls: list, metric: str, log, use_combo: bool = False) -> tuple[list, list]:
    """Run all tool calls and return messages + metric values."""
    results, values = [], []
    for tc in tool_calls:
        if tc.function.name != "evaluate_factor":
            continue
        try:
            args = json.loads(tc.function.arguments)
        except Exception:
            continue
        name = args.get("factor_name", "factor")
        expr = args.get("factor_expr", "")
        if use_combo:
            val, status = combo_backtest(name, expr, metric)
        else:
            val, status = backtest(name, expr, metric)
        values.append(val)
        msg = f"Factor: {name} | expr: {expr} | {metric}: {val} | {status}"
        log(msg)
        content = (
            f'success: Evaluated factor "{name}" with expression "{expr}", {metric}={val}'
            if status == "success"
            else f'failed: Factor {name} expression "{expr}". Reason: {status}'
        )
        results.append({
            "tool_call_id": tc.id,
            "role": "tool",
            "name": "evaluate_factor",
            "content": content,
        })
    return results, values


def evolve_one(seed: dict, n_turns: int, metric: str, log, ensemble: list = None, use_combo: bool = False):
    """Run multi-turn evolution for a single seed. Returns list of all tried factors.

    Args:
        ensemble: list of OTHER seeds in the ensemble (not including this seed).
                  Passed as context so Qwen can evolve toward low-correlation variants.
        use_combo: if True, use combo_backtest (LGBModel multi-factor) instead of single-factor proxy.
    """
    seed_name = seed["name"]
    seed_expr = seed["expr"]

    # Step 1: backtest seed
    if use_combo:
        init_val, status = combo_backtest(seed_name, seed_expr, metric)
    else:
        init_val, status = backtest(seed_name, seed_expr, metric)
    log(f"\n{'='*60}")
    log(f"SEED: {seed_name}")
    log(f"EXPR: {seed_expr}")
    log(f"INIT {metric}: {init_val} ({status})")

    if status != "success" or np.isnan(init_val):
        log("Skipping — seed backtest failed")
        return []

    # Build ensemble context string
    ensemble_ctx = ""
    if ensemble:
        ensemble_ctx = "\n## The other factors in the ensemble (DO NOT modify these):\n"
        for i, e in enumerate(ensemble, 1):
            ensemble_ctx += f"Factor {i}: {e['name']}\nExpression: {e['expr']}\n\n"

    # Build initial messages
    user_content = (
        f"You are evolving the TARGET factor below. It is part of a 3-factor ensemble "
        f"used in a LightGBM model on Vietnam stocks (test period 2022-2025).\n"
        f"{ensemble_ctx}"
        f"## TARGET factor to evolve:\n"
        f"Name: {seed_name}\n"
        f"Expression: {seed_expr}\n"
        f"Proxy {metric}: {init_val}\n\n"
        f"Propose 2-3 improved variants of the TARGET factor. "
        f"Call evaluate_factor for each. Aim for higher {metric} and low correlation with the other factors."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    all_factors = [{"name": seed_name, "expr": seed_expr, metric: init_val, "turn": 0}]
    best_val = init_val

    for turn in range(1, n_turns + 1):
        log(f"\n--- Turn {turn}/{n_turns} ---")
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=3000,
                temperature=0.7,
                top_p=0.9,
                tools=[TOOL_SCHEMA],
                tool_choice="auto",
            )
        except Exception as e:
            log(f"LLM error: {e}")
            break

        msg = resp.choices[0].message
        log(f"Model thinking done. Tool calls: {len(msg.tool_calls or [])}")

        if not msg.tool_calls:
            log("No tool calls — stopping early")
            break

        # Append assistant message
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        # Execute tool calls
        tool_results, vals = execute_tool_calls(msg.tool_calls, metric, log, use_combo=use_combo)
        messages.extend(tool_results)

        # Track results
        for tc, val in zip(msg.tool_calls, vals):
            try:
                args = json.loads(tc.function.arguments)
                factor_entry = {
                    "name": args.get("factor_name", ""),
                    "expr": args.get("factor_expr", ""),
                    metric: val,
                    "turn": turn,
                }
                all_factors.append(factor_entry)
                if not np.isnan(val) and val > best_val:
                    best_val = val
                    log(f"  *** New best {metric}: {val} ***")
            except Exception:
                pass

        # Add turn summary to prompt
        valid_vals = [v for v in vals if not np.isnan(v)]
        if valid_vals:
            messages.append({
                "role": "user",
                "content": (
                    f"Turn {turn} results: best {metric} this turn = {max(valid_vals):.4f}, "
                    f"overall best = {best_val:.4f} (seed was {init_val:.4f}). "
                    f"Continue evolving — propose 2-3 more improved variants."
                ),
            })

    log(f"\nSeed {seed_name}: init={init_val:.4f}, best={best_val:.4f}, improved={best_val > init_val}")
    return all_factors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to evolve")
    parser.add_argument("--turns", type=int, default=3, help="Evolution turns per seed")
    parser.add_argument("--metric", type=str, default="Information_Ratio_with_cost", help="Metric to optimize")
    parser.add_argument("--combo", action="store_true", help="Use combo LGBModel evaluator (port 8003)")
    parser.add_argument("--evolve_slot", type=int, default=None,
                        help="Which factor slot to evolve (1,2,3). Others become fixed. Requires --combo.")
    args = parser.parse_args()

    # If combo mode, set up fixed factors and adjust metric
    use_combo = args.combo
    if use_combo:
        args.metric = "IR_without_cost"  # combo metric

    # Load seeds
    with open(SEEDS_FILE) as f:
        all_seeds = [json.loads(l) for l in f if l.strip()]

    # "Keep 2, Evolve 1" setup
    global FIXED_FACTORS
    if use_combo and args.evolve_slot is not None and len(all_seeds) >= 3:
        slot = args.evolve_slot - 1  # 0-indexed
        FIXED_FACTORS = {s["name"]: s["expr"] for i, s in enumerate(all_seeds[:3]) if i != slot}
        selected = [all_seeds[slot]]
        print(f"COMBO MODE: Keep 2, Evolve slot {args.evolve_slot}")
        print(f"  Fixed: {list(FIXED_FACTORS.keys())}")
        print(f"  Evolving: {selected[0]['name']}")
    else:
        selected = all_seeds[:args.seeds]

    print(f"Evolving {len(selected)} seeds, {args.turns} turns each, metric={args.metric}")
    if use_combo:
        print(f"Using COMBO evaluator (LGBModel multi-factor, port 8003)")
    print(f"Results → {RESULTS_DIR}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    summary = []

    for i, seed in enumerate(selected):
        log_file = RESULTS_DIR / f"seed_{i+1:03d}_{seed['name']}_{run_id}.log"
        with open(log_file, "w") as lf:
            def log(msg, _lf=lf):
                print(msg)
                _lf.write(msg + "\n")
                _lf.flush()

            # Pass the other seeds as ensemble context
            ensemble_others = [s for s in all_seeds[:3] if s["name"] != seed["name"]]
            factors = evolve_one(seed, args.turns, args.metric, log,
                                ensemble=ensemble_others, use_combo=use_combo)
            all_results.extend(factors)

            if factors:
                valid = [f for f in factors if not np.isnan(f.get(args.metric, np.nan))]
                if valid:
                    best = max(valid, key=lambda x: x.get(args.metric, -np.inf))
                    summary.append({
                        "seed_name": seed["name"],
                        "seed_expr": seed["expr"],
                        f"seed_{args.metric}": factors[0].get(args.metric, np.nan),
                        f"best_{args.metric}": best.get(args.metric, np.nan),
                        "best_expr": best["expr"],
                        "best_turn": best["turn"],
                        "n_tried": len(factors),
                    })

        time.sleep(1)

    # Save all results
    results_file = RESULTS_DIR / f"all_factors_{run_id}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    summary_file = RESULTS_DIR / f"summary_{run_id}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {len(summary)} seeds evolved")
    print(f"{'='*60}")
    improved = [s for s in summary if s[f"best_{args.metric}"] > s[f"seed_{args.metric}"]]
    print(f"Improved: {len(improved)}/{len(summary)}")
    for s in sorted(summary, key=lambda x: x[f"best_{args.metric}"], reverse=True)[:10]:
        delta = s[f"best_{args.metric}"] - s[f"seed_{args.metric}"]
        print(f"  {s['seed_name']:12s} seed={s[f'seed_{args.metric}']:.4f} → best={s[f'best_{args.metric}']:.4f} (+{delta:.4f}) turn={s['best_turn']}")
    print(f"\nSaved: {results_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
