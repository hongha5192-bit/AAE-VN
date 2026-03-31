#!/usr/bin/env python3
"""Pretty-print AlphaAgentEvo training logs for notebook stdout.

Reads raw trainer output on stdin, emits concise human-readable progress to stdout,
and optionally mirrors the concise stream to a side log file.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"\bstep:(\d+)\b")
PROGRESS_RE = re.compile(r"Training Progress:\s*(\d+)%.*?(\d+)/(\d+)")
CAPTURE_RE = re.compile(r"Capturing batches.*?(\d+)/(\d+)")
FLOAT_TEMPLATE = r"{key}:([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "\n")


def extract_float(line: str, key: str) -> float | None:
    match = re.search(FLOAT_TEMPLATE.format(key=re.escape(key)), line)
    return float(match.group(1)) if match else None


def extract_val_paper_metric(line: str, metric_name: str) -> float | None:
    match = re.search(
        FLOAT_TEMPLATE.format(key=rf"val-paper/[^/]+/{re.escape(metric_name)}"),
        line,
    )
    return float(match.group(1)) if match else None


def compact_text(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class PrettyPrinter:
    def __init__(self, total_steps: int | None = None, pretty_log: Path | None = None):
        self.total_steps = total_steps
        self.pretty_log = pretty_log
        self.last_progress: tuple[int, int, int] | None = None
        self.last_capture: tuple[int, int] | None = None
        self.last_step: int | None = None
        self.last_step_line: str | None = None
        self.validation_active = False
        self.checkpoint_active = False
        self.last_validation_line: str | None = None
        self.last_checkpoint_line: str | None = None

    def emit(self, line: str) -> None:
        print(line, flush=True)
        if self.pretty_log is not None:
            self.pretty_log.parent.mkdir(parents=True, exist_ok=True)
            with self.pretty_log.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def handle(self, raw_line: str) -> None:
        for clean_line in strip_ansi(raw_line).splitlines():
            self._handle_clean_line(clean_line.strip())

    def _handle_clean_line(self, line: str) -> None:
        if not line:
            return

        if (
            line.startswith("[train-entry]")
            or line.startswith("[heartbeat]")
            or line.startswith("[train-exit]")
            or line.startswith("[notebook-live]")
        ):
            self.emit(line)
            return

        if "[factor-live]" in line or "[response-live]" in line:
            marker = "[factor-live]" if "[factor-live]" in line else "[response-live]"
            self.emit(line[line.index(marker) :])
            return

        if "Training Progress:" in line:
            match = PROGRESS_RE.search(line)
            if match:
                pct, cur, total = (int(match.group(i)) for i in range(1, 4))
                value = (pct, cur, total)
                if value != self.last_progress:
                    self.last_progress = value
                    self.emit(f"[progress] train_step={cur}/{total} pct={pct}")
            return

        if "Saving tensorboard log to " in line:
            self.emit("[phase] validation_setup tensorboard_log_open")
            return

        if "test_gen_batch meta info:" in line:
            if not self.validation_active:
                self.validation_active = True
                phase_line = "[phase] validation_start"
                if phase_line != self.last_validation_line:
                    self.last_validation_line = phase_line
                    self.emit(phase_line)
            return

        if "Capturing batches" in line:
            match = CAPTURE_RE.search(line)
            if match:
                cur, total = (int(match.group(i)) for i in range(1, 3))
                value = (cur, total)
                if value != self.last_capture:
                    self.last_capture = value
                    avail_mem = None
                    mem_match = re.search(r"avail_mem=([^):]+)", line)
                    if mem_match:
                        avail_mem = mem_match.group(1)
                    extra = f" avail_mem={avail_mem}" if avail_mem else ""
                    self.emit(f"[warmup] capture_batches={cur}/{total}{extra}")
            return

        if "step:" in line and "critic/rewards/mean:" in line:
            self._emit_step_summary(line)
            return

        if "Final validation metrics:" in line:
            if self.validation_active:
                self.emit("[phase] validation_done")
                self.validation_active = False
            self.emit(compact_text(line, limit=260))
            return

        if "Saved model to " in line or "Saved extra_state to " in line or "Saved hf_model to " in line:
            self._emit_checkpoint_marker(line)
            return

        important_passthroughs = (
            "Saved model to ",
            "Saved hf_model",
            "Saved extra_state",
            "Saving tensorboard log",
            "RayTaskError",
            "ActorDiedError",
            "Traceback",
            "RuntimeError:",
            "ValueError:",
            "KeyError:",
            "AssertionError:",
            "Error executing job",
            "TRAIN SUMMARY",
            "TRAIN FAILURE CONTEXT",
        )
        if any(token in line for token in important_passthroughs):
            self.emit(compact_text(line, limit=260))

    def _emit_step_summary(self, line: str) -> None:
        step_match = STEP_RE.search(line)
        if not step_match:
            return

        step = int(step_match.group(1))
        reward = extract_float(line, "critic/rewards/mean")
        val_reward = extract_float(line, "val-core/alphaagentevo/reward/mean@3")
        entropy = extract_float(line, "actor/entropy")
        response_length = extract_float(line, "response_length/mean")
        step_time = extract_float(line, "timing_s/step")
        throughput = extract_float(line, "perf/throughput")
        score_mean = extract_float(line, "critic/score/mean")
        vr = extract_val_paper_metric(line, "vr")
        pass_at_3 = extract_val_paper_metric(line, "pass@3")
        pass_at_5 = extract_val_paper_metric(line, "pass@5")
        beat_rate = extract_val_paper_metric(line, "beat_rate")
        best_metric_mean = extract_val_paper_metric(line, "best_metric_mean")

        total = self.total_steps
        suffix = f"/{total}" if total else ""
        parts = [f"[step {step}{suffix}]"]
        if reward is not None:
            parts.append(f"reward={reward:.4f}")
        if val_reward is not None:
            parts.append(f"val={val_reward:.4f}")
        if score_mean is not None:
            parts.append(f"score={score_mean:.4f}")
        if entropy is not None:
            parts.append(f"entropy={entropy:.3f}")
        if response_length is not None:
            parts.append(f"resp_len={response_length:.1f}")
        if step_time is not None:
            parts.append(f"step_s={step_time:.2f}")
        if throughput is not None:
            parts.append(f"tok/s={throughput:.1f}")

        summary = " ".join(parts)
        if step == self.last_step and summary == self.last_step_line:
            return
        self.last_step = step
        self.last_step_line = summary
        self.emit(summary)

        if any(value is not None for value in (val_reward, vr, pass_at_3, pass_at_5, beat_rate, best_metric_mean)):
            if self.validation_active:
                self.emit("[phase] validation_done")
                self.validation_active = False
            val_parts = [f"[validation {step}{suffix}]"]
            if val_reward is not None:
                val_parts.append(f"reward={val_reward:.4f}")
            if vr is not None:
                val_parts.append(f"vr={vr:.3f}")
            if pass_at_3 is not None:
                val_parts.append(f"pass@3={pass_at_3:.3f}")
            if pass_at_5 is not None:
                val_parts.append(f"pass@5={pass_at_5:.3f}")
            if beat_rate is not None:
                val_parts.append(f"beat_rate={beat_rate:.3f}")
            if best_metric_mean is not None:
                val_parts.append(f"best_metric_mean={best_metric_mean:.4f}")
            self.emit(" ".join(val_parts))

    def _emit_checkpoint_marker(self, line: str) -> None:
        step_match = re.search(r"global_step_(\d+)", line)
        step_text = step_match.group(1) if step_match else "?"

        if "Saved model to " in line:
            label = f"[phase] checkpoint_model_saved step={step_text}"
        elif "Saved extra_state to " in line:
            label = f"[phase] checkpoint_extra_saved step={step_text}"
        else:
            label = f"[phase] checkpoint_hf_saved step={step_text}"

        if not self.checkpoint_active:
            self.emit(f"[phase] checkpoint_start step={step_text}")
            self.checkpoint_active = True

        if label != self.last_checkpoint_line:
            self.last_checkpoint_line = label
            self.emit(label)

        if "Saved hf_model to " in line:
            self.emit(f"[phase] checkpoint_done step={step_text}")
            self.checkpoint_active = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--pretty-log", default=os.environ.get("TRAIN_PRETTY_LOG", ""))
    args = parser.parse_args()

    pretty_log = Path(args.pretty_log) if args.pretty_log else None
    printer = PrettyPrinter(total_steps=args.total_steps, pretty_log=pretty_log)

    for raw_line in sys.stdin:
        printer.handle(raw_line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
