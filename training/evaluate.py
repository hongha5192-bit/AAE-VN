"""Evaluate trained AlphaAgentEvo checkpoints on val/test seed sets.

Metrics (matching paper):
  - VR (Valid Ratio): % of generated factors that are syntactically valid
  - pass@T: fraction of seeds where at least one evolved factor beats max(0, seed_ir)
  - Mean best IR improvement over seed
"""

import argparse
import json
import os
import re
import time
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.factor_tool import FactorTool

FACTOR_RE = re.compile(r"^Factor:\s*(.+)$", re.MULTILINE)
EXPR_RE = re.compile(r"^Expression:\s*(.+)$", re.MULTILINE)
BASELINE_RE = re.compile(r"^Baseline IR:\s*([+-]?\d+(?:\.\d+)?)$", re.MULTILINE)


def _as_prompt_list(raw_prompt):
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


def _first_message(prompt, role: str) -> str:
    for message in _as_prompt_list(prompt):
        if message.get("role") == role:
            return str(message.get("content", "")).strip()
    return ""


def _extract_seed_name(row: pd.Series, idx: int) -> str:
    value = row.get("seed_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    user_msg = _first_message(row.get("prompt"), "user")
    match = FACTOR_RE.search(user_msg)
    if match:
        return match.group(1).strip()
    return f"seed_{idx}"


def _extract_seed_expr(row: pd.Series) -> str:
    value = row.get("seed_expr")
    if isinstance(value, str) and value.strip():
        return value.strip()
    user_msg = _first_message(row.get("prompt"), "user")
    match = EXPR_RE.search(user_msg)
    if match:
        return match.group(1).strip()
    return ""


def _extract_seed_ir(row: pd.Series) -> float:
    value = row.get("seed_ir")
    if value is not None and not pd.isna(value):
        try:
            return float(value)
        except Exception:
            pass
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        try:
            return float(reward_model.get("ground_truth", 0.0))
        except Exception:
            pass
    user_msg = _first_message(row.get("prompt"), "user")
    match = BASELINE_RE.search(user_msg)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            pass
    return 0.0


def _normalize_eval_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "seed_name" not in out.columns:
        out["seed_name"] = [_extract_seed_name(row, idx) for idx, row in out.iterrows()]
    if "seed_expr" not in out.columns:
        out["seed_expr"] = [_extract_seed_expr(row) for _, row in out.iterrows()]
    if "seed_ir" not in out.columns:
        out["seed_ir"] = [_extract_seed_ir(row) for _, row in out.iterrows()]
    if "prompt" not in out.columns:
        raise ValueError("Input parquet must contain column 'prompt'")
    return out


def _clip_text(text: str, max_log_chars: int) -> str:
    if max_log_chars <= 0 or len(text) <= max_log_chars:
        return text
    return f"{text[:max_log_chars]}\n... <truncated {len(text) - max_log_chars} chars>"


def _guess_tokenizer_fallback(base_model_path: str) -> str | None:
    """Try to resolve original model id from a local checkpoint dir."""
    cfg_path = Path(base_model_path) / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return None
    for key in ("_name_or_path", "name_or_path"):
        val = cfg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def load_model(base_model_path: str, checkpoint_path: str | None, device: str = "cuda"):
    """Load base model + optional LoRA adapter."""
    print(f"Loading tokenizer from {base_model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left",
        )
    except Exception as exc:
        fallback = _guess_tokenizer_fallback(base_model_path)
        # Common failure in exported local checkpoints on this stack:
        # tokenizer_config has extra_special_tokens in incompatible format.
        if fallback and fallback != base_model_path:
            print(f"Tokenizer load failed from local dir: {exc}")
            print(f"Retrying tokenizer from base id: {fallback}")
            tokenizer = AutoTokenizer.from_pretrained(
                fallback,
                trust_remote_code=True,
                padding_side="left",
            )
        else:
            raise

    # Swap chat template for Thinking model (same as train.py)
    if "Thinking" in base_model_path or "thinking" in base_model_path:
        base_name = base_model_path.replace("-Thinking-2507", "").replace("-thinking", "")
        base_tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        tokenizer.chat_template = base_tok.chat_template
        del base_tok

    print(f"Loading model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if checkpoint_path:
        print(f"Loading LoRA adapter from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()  # Merge for faster inference
        print("LoRA merged.")

    model.eval()
    return model, tokenizer


def build_tool_schema():
    """Build the tool schema for evaluate_factor (matching train.py)."""
    return {
        "type": "function",
        "function": {
            "name": "evaluate_factor",
            "description": "Evaluate a factor expression by backtesting against historical Vietnam stock market data. Returns the Information Ratio (IR), mean IC, and success status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "factor_name": {
                        "type": "string",
                        "description": "A descriptive name for the factor"
                    },
                    "factor_expr": {
                        "type": "string",
                        "description": "The factor expression using available variables and operators"
                    }
                },
                "required": ["factor_name", "factor_expr"]
            }
        }
    }


def run_inference(
    model,
    tokenizer,
    messages,
    tool_schema,
    max_turns=3,
    max_new_tokens=5000,
    max_tool_calls_per_turn=4,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    print_io=False,
    max_log_chars=4000,
):
    """Run multi-turn inference with tool calling.

    Returns:
        all_results: list of {success, ir, factor_expr} from each tool call
        full_messages: the complete conversation
    """
    factor_tool = FactorTool()
    all_results = []
    full_messages = list(messages)

    for turn in range(max_turns):
        # Apply chat template with tools
        text = tokenizer.apply_chat_template(
            full_messages,
            tools=[tool_schema],
            tokenize=False,
            add_generation_prompt=True,
        )
        if print_io:
            print(f"\n[turn {turn+1}] input")
            print(_clip_text(text, max_log_chars))

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(do_sample),
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            outputs = model.generate(**gen_kwargs)

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        if print_io:
            print(f"\n[turn {turn+1}] output")
            print(_clip_text(response_text, max_log_chars))

        # Parse tool calls from response
        tool_calls = parse_tool_calls(response_text)
        if len(tool_calls) > max_tool_calls_per_turn:
            if print_io:
                print(
                    f"[turn {turn+1}] parsed {len(tool_calls)} tool calls, "
                    f"keeping first {max_tool_calls_per_turn}"
                )
            tool_calls = tool_calls[:max_tool_calls_per_turn]

        if not tool_calls:
            # No tool calls — model is done
            full_messages.append({"role": "assistant", "content": response_text})
            break

        # Execute each tool call
        assistant_msg = {
            "role": "assistant",
            "content": response_text.split("<tool_call>")[0].strip() if "<tool_call>" in response_text else "",
            "tool_calls": []
        }

        tool_responses = []
        for i, tc in enumerate(tool_calls):
            call_id = f"call_{turn}_{i}"
            if print_io:
                print(f"[turn {turn+1}] tool_call[{i+1}] {json.dumps(tc, ensure_ascii=False)}")
            assistant_msg["tool_calls"].append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "evaluate_factor",
                    "arguments": json.dumps(tc)
                }
            })

            # Execute via API
            result = factor_tool.evaluate(
                factor_name=tc.get("factor_name", "unnamed"),
                factor_expr=tc.get("factor_expr", ""),
            )
            all_results.append({
                "success": result.get("success", False),
                "ir": result.get("ir", 0.0),
                "factor_expr": tc.get("factor_expr", ""),
                "factor_name": tc.get("factor_name", ""),
                "turn": turn,
            })
            if print_io:
                print(f"[turn {turn+1}] tool_result[{i+1}] {json.dumps(result, ensure_ascii=False)}")

            tool_responses.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result),
            })

        full_messages.append(assistant_msg)
        full_messages.extend(tool_responses)

    return all_results, full_messages


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool-call arguments from model output.

    Supports:
    - Tagged blocks: <tool_call>{...}</tool_call>
    - Naked JSON objects in plain text
    - OpenAI-style wrappers: {"name":"evaluate_factor","arguments":{...}}
    - Nested function wrapper: {"function":{"name":"evaluate_factor","arguments":...}}
    """
    calls: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _append_args(args):
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return
        if not isinstance(args, dict):
            return
        if "factor_expr" not in args:
            return
        key = (str(args.get("factor_name", "")), str(args.get("factor_expr", "")))
        if key in seen:
            return
        seen.add(key)
        calls.append(args)

    def _append_from_obj(obj):
        if not isinstance(obj, dict):
            return
        if obj.get("name") == "evaluate_factor":
            _append_args(obj.get("arguments", {}))
            return
        fn = obj.get("function")
        if isinstance(fn, dict) and fn.get("name") == "evaluate_factor":
            _append_args(fn.get("arguments", {}))
            return
        _append_args(obj)

    # 1) Tagged <tool_call> blocks
    pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
    for match in pattern.finditer(text):
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        _append_from_obj(obj)

    # 2) Naked JSON scanning (robust against extra text)
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, step = dec.raw_decode(text[i:])
        except Exception:
            i += 1
            continue
        _append_from_obj(obj)
        i += max(step, 1)

    return calls


def evaluate_checkpoint(
    model,
    tokenizer,
    seeds_df,
    tool_schema,
    max_turns=3,
    max_new_tokens=5000,
    max_tool_calls_per_turn=4,
    do_sample=False,
    temperature=0.7,
    top_p=0.9,
    print_io=False,
    max_log_chars=4000,
):
    """Evaluate a model on a set of seeds.

    Returns dict with metrics: vr, pass_at_3, pass_at_5, per-seed details
    """
    results = []
    seeds_df = _normalize_eval_dataframe(seeds_df)

    for idx, row in seeds_df.iterrows():
        seed_name = row["seed_name"]
        seed_expr = row["seed_expr"]
        seed_ir = row["seed_ir"]
        messages = row["prompt"]  # Already a list of message dicts

        print(f"  [{idx+1}/{len(seeds_df)}] {seed_name} (seed IR={seed_ir:.4f})...", end="", flush=True)
        t0 = time.time()

        try:
            all_tool_results, _ = run_inference(
                model, tokenizer, messages, tool_schema,
                max_turns=max_turns,
                max_new_tokens=max_new_tokens,
                max_tool_calls_per_turn=max_tool_calls_per_turn,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                print_io=print_io,
                max_log_chars=max_log_chars,
            )
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "seed_name": seed_name,
                "seed_ir": seed_ir,
                "n_calls": 0,
                "n_valid": 0,
                "best_ir": None,
                "beat_seed": False,
                "all_irs": [],
                "error": str(e),
            })
            continue

        n_calls = len(all_tool_results)
        valid = [r for r in all_tool_results if r["success"]]
        n_valid = len(valid)
        irs = [r["ir"] for r in valid]
        best_ir = max(irs) if irs else None
        baseline = max(0.0, seed_ir)
        beat_seed = best_ir is not None and best_ir > baseline

        elapsed = time.time() - t0

        # Compute pass at different turn counts
        irs_by_turn = {}
        for r in all_tool_results:
            t = r["turn"]
            if r["success"]:
                irs_by_turn.setdefault(t, []).append(r["ir"])

        # Cumulative best IR at each turn
        cum_best = None
        pass_at = {}
        for t in range(max_turns):
            if t in irs_by_turn:
                turn_best = max(irs_by_turn[t])
                cum_best = max(cum_best, turn_best) if cum_best is not None else turn_best
            pass_at[t] = cum_best is not None and cum_best > baseline

        status = "BEAT" if beat_seed else "miss"
        ir_str = f"{best_ir:.4f}" if best_ir is not None else "N/A"
        print(f" {status} | best_ir={ir_str} | {n_valid}/{n_calls} valid | {elapsed:.1f}s")

        results.append({
            "seed_name": seed_name,
            "seed_ir": seed_ir,
            "n_calls": n_calls,
            "n_valid": n_valid,
            "best_ir": best_ir,
            "beat_seed": beat_seed,
            "pass_at": pass_at,
            "all_irs": irs,
            "all_results": all_tool_results,
        })

    # Compute aggregate metrics
    n_seeds = len(results)
    vr_per_seed = [r["n_valid"] / r["n_calls"] if r["n_calls"] > 0 else 0 for r in results]
    vr = sum(vr_per_seed) / len(vr_per_seed) if vr_per_seed else 0

    pass_at_3 = sum(1 for r in results if r.get("pass_at", {}).get(2, False)) / n_seeds if n_seeds > 0 else 0
    # pass_at_5 not applicable since we only do 3 turns, use overall beat rate
    beat_rate = sum(1 for r in results if r["beat_seed"]) / n_seeds if n_seeds > 0 else 0

    # Mean IR improvement for seeds that were beaten
    beaten = [r for r in results if r["beat_seed"]]
    mean_ir_improvement = 0
    if beaten:
        improvements = [r["best_ir"] - max(0, r["seed_ir"]) for r in beaten]
        mean_ir_improvement = sum(improvements) / len(improvements)

    metrics = {
        "n_seeds": n_seeds,
        "vr": round(vr, 4),
        "pass_at_3": round(pass_at_3, 4),
        "beat_rate": round(beat_rate, 4),
        "mean_ir_improvement": round(mean_ir_improvement, 4),
        "n_beaten": len(beaten),
    }

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaAgentEvo checkpoints")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B-MLX-bf16")
    parser.add_argument("--checkpoint", default=None, help="Path to LoRA checkpoint dir")
    parser.add_argument("--data", required=True, help="Path to parquet file (val or test)")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--max-tool-calls-per-turn", type=int, default=4)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling during generation (default: greedy)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--print-io", action="store_true", help="Print full input/output text per turn")
    parser.add_argument("--max-log-chars", type=int, default=4000, help="Clip printed I/O text length per turn")
    parser.add_argument("--output-dir", default=None, help="Directory to save eval json")
    parser.add_argument("--label", default="", help="Label for this evaluation run")
    args = parser.parse_args()

    label = args.label or (Path(args.checkpoint).name if args.checkpoint else "base")
    print(f"\n{'='*60}")
    print(f"EVALUATION: {label}")
    print(f"{'='*60}")
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint or 'NONE (base model)'}")
    print(f"Data: {args.data}")
    print(f"Max turns: {args.max_turns}")
    print(f"Max tool calls/turn: {args.max_tool_calls_per_turn}")
    print(f"Sampling: {'on' if args.do_sample else 'off'}")
    print(f"Print I/O: {'yes' if args.print_io else 'no'}")
    print()

    # Load data
    seeds_df = pd.read_parquet(args.data)
    print(f"Loaded {len(seeds_df)} seeds")

    # Load model
    model, tokenizer = load_model(args.base_model, args.checkpoint)

    # Build tool schema
    tool_schema = build_tool_schema()

    # Evaluate
    t0 = time.time()
    metrics, details = evaluate_checkpoint(
        model, tokenizer, seeds_df, tool_schema,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        max_tool_calls_per_turn=args.max_tool_calls_per_turn,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        print_io=args.print_io,
        max_log_chars=args.max_log_chars,
    )
    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")
    print(f"  Seeds:              {metrics['n_seeds']}")
    print(f"  Valid Ratio (VR):   {metrics['vr']:.4f}")
    print(f"  Pass@3:             {metrics['pass_at_3']:.4f}")
    print(f"  Beat Rate (overall):{metrics['beat_rate']:.4f}")
    print(f"  Seeds beaten:       {metrics['n_beaten']}/{metrics['n_seeds']}")
    print(f"  Mean IR improvement:{metrics['mean_ir_improvement']:.4f}")
    print(f"  Total time:         {elapsed:.1f}s ({elapsed/len(seeds_df):.1f}s/seed)")
    print()

    # Per-seed details
    print("Per-seed results:")
    print(f"  {'Seed':<20} {'Seed IR':>8} {'Best IR':>8} {'Valid':>6} {'Beat':>5}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*6} {'-'*5}")
    for r in details:
        best = f"{r['best_ir']:.4f}" if r['best_ir'] is not None else "N/A"
        valid = f"{r['n_valid']}/{r['n_calls']}" if r['n_calls'] > 0 else "0/0"
        beat = "YES" if r['beat_seed'] else "no"
        print(f"  {r['seed_name']:<20} {r['seed_ir']:>8.4f} {best:>8} {valid:>6} {beat:>5}")

    # Save results
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    if output_dir is None:
        default_output = PROJECT_ROOT / "eval_results"
        if os.access(str(default_output.parent), os.W_OK):
            output_dir = default_output
        else:
            output_dir = Path("/kaggle/working/aae_v2/eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_name = Path(args.data).stem
    out_file = output_dir / f"eval_{label}_{data_name}.json"
    with open(out_file, "w") as f:
        # Convert non-serializable items
        save_details = []
        for r in details:
            save_r = {k: v for k, v in r.items() if k != "all_results"}
            save_r["pass_at"] = {str(k): v for k, v in r.get("pass_at", {}).items()}
            save_details.append(save_r)
        json.dump({"metrics": metrics, "details": save_details, "config": vars(args)}, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    return metrics


if __name__ == "__main__":
    main()
