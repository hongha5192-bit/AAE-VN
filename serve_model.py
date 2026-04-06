"""
OpenAI-compatible API server for AlphaAgentEvo fine-tuned Qwen3 model.
Exposes POST /v1/chat/completions so AlphaAgent can use it instead of ChatGPT.

Usage:
    conda activate alphaevo
    python AAE-VN/serve_model.py --model /home/dc_analyst/Ha/AlphaAgent --port 8100
"""

import argparse
import json
import re
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

# ---------------------------------------------------------------------------
# Request / Response schemas (OpenAI-compatible subset)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-alpha"
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    stream: bool = False
    tools: list | None = None
    tool_choice: str | dict = "auto"
    seed: int | None = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL = None
TOKENIZER = None


def load_model(model_path: str):
    global MODEL, TOKENIZER
    print(f"Loading tokenizer from {model_path}...")
    TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    print(f"Loading model from {model_path}...")
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    MODEL.eval()
    print("Model loaded and ready.")


# ---------------------------------------------------------------------------
# Tool-call parsing (Qwen3 format)
# ---------------------------------------------------------------------------

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


def _parse_tool_calls(text: str):
    """Extract tool calls from Qwen3 <tool_call>...</tool_call> tags."""
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Try single-quote → double-quote conversion
            try:
                obj = json.loads(raw.replace("'", '"'))
            except Exception:
                continue
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": obj.get("name", ""),
                "arguments": json.dumps(obj.get("arguments", obj.get("parameters", {}))),
            },
        })
    return calls


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _repair_json(candidate: str) -> str:
    """Attempt to repair common JSON formatting issues from model output."""
    import json as _json

    # Try as-is first
    try:
        _json.loads(candidate)
        return candidate
    except Exception:
        pass

    repaired = candidate

    # Convert single-quoted keys/values to double quotes
    # Replace 'key': with "key":
    repaired = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'(\s*:)", r'"\1"\2', repaired)
    # Replace ': 'value' with ": "value"
    repaired = re.sub(r"(:\s*)'([^'\\]*(?:\\.[^'\\]*)*)'(\s*[,}\]])", r'\1"\2"\3', repaired)

    # Remove trailing commas before } or ]
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

    # Replace Python True/False/None with JSON equivalents
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)

    try:
        _json.loads(repaired)
        return repaired
    except Exception:
        pass

    return candidate


def _find_json_end(text: str, start: int, opener: str, closer: str) -> int:
    """Find the index of the matching closer for opener at text[start], counting nesting."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _extract_json(text: str) -> str:
    """If response contains JSON object/array, extract just that part.
    Handles cases where model outputs reasoning text before/after the JSON.
    Also repairs common formatting issues (single quotes, trailing commas).
    Uses bracket counting to find the real end of the JSON object.
    """
    import json as _json

    def _try(candidate: str) -> str | None:
        c = candidate.strip()
        repaired = _repair_json(c)
        try:
            _json.loads(repaired)
            return repaired
        except Exception:
            return None

    # Look for ```json ... ``` block first (most reliable)
    md = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if md:
        result = _try(md.group(1))
        if result:
            return result

    # Find first { and use bracket counting to locate its matching }
    start = text.find("{")
    if start != -1:
        end = _find_json_end(text, start, "{", "}")
        if end != -1:
            result = _try(text[start:end+1])
            if result:
                return result

    # Find first [ and use bracket counting
    start = text.find("[")
    if start != -1:
        end = _find_json_end(text, start, "[", "]")
        if end != -1:
            result = _try(text[start:end+1])
            if result:
                return result

    # Last resort: return the raw stripped text (caller will get parse error)
    return text


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AlphaAgentEvo Model Server")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "qwen3-alpha", "object": "model", "created": int(time.time()), "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported — set stream=false in AlphaAgent config")

    # Convert messages to dicts for apply_chat_template
    messages = []
    for m in request.messages:
        msg = {"role": m.role}
        if m.content is not None:
            msg["content"] = m.content
        if m.tool_calls is not None:
            msg["tool_calls"] = m.tool_calls
        if m.tool_call_id is not None:
            msg["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            msg["name"] = m.name
        messages.append(msg)

    # Build tools list for template
    tools = request.tools if request.tools else None

    try:
        text = TOKENIZER.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback: no tools in template
        text = TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = TOKENIZER(text, return_tensors="pt").to(MODEL.device)
    input_len = inputs["input_ids"].shape[1]

    # Cap at 3000 tokens: at ~25 tok/s = ~120s, well under 5-min timeout.
    # Qwen3 thinking uses ~1500-2000 tokens, leaving ~1000-1500 for JSON answer.
    MAX_OUTPUT_TOKENS = min(request.max_tokens, 3000)

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_TOKENS,
            do_sample=request.temperature > 0,
            temperature=request.temperature if request.temperature > 0 else 1.0,
            top_p=request.top_p,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    generated = outputs[0][input_len:]
    raw_text = TOKENIZER.decode(generated, skip_special_tokens=True)
    clean_text = _strip_thinking(raw_text)

    # Check for tool calls
    tool_calls = _parse_tool_calls(clean_text)

    if tool_calls:
        # Remove tool_call blocks from content
        content = TOOL_CALL_RE.sub("", clean_text).strip() or None
        message = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
        finish_reason = "tool_calls"
    else:
        # Extract JSON if model prepended reasoning text before the JSON
        final_text = _extract_json(clean_text)
        message = {"role": "assistant", "content": final_text}
        finish_reason = "stop"

    prompt_tokens = input_len
    completion_tokens = len(generated)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve AlphaAgentEvo model as OpenAI-compatible API")
    parser.add_argument("--model", default="/home/dc_analyst/Ha/AlphaAgent", help="Path to merged model")
    parser.add_argument("--port", type=int, default=8100, help="Port to serve on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    args = parser.parse_args()

    load_model(args.model)

    uvicorn.run(app, host=args.host, port=args.port)
