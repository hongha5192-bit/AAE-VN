#!/usr/bin/env python3
"""Parse AlphaAgentEvo/VERL train logs and plot step-aligned metrics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"\bstep:(\d+)\b")
FLOAT_TEMPLATE = r"{key}:([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
PRETTY_STEP_RE = re.compile(r"\[step\s+(\d+)(?:/\d+)?\]")
PRETTY_VAL_RE = re.compile(r"\[validation\s+(\d+)(?:/\d+)?\]")


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


def extract_pretty_float(line: str, key: str) -> float | None:
    match = re.search(rf"\b{re.escape(key)}=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", line)
    return float(match.group(1)) if match else None


def parse_step_metrics(log_text: str) -> list[dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}

    for raw_line in strip_ansi(log_text).splitlines():
        pretty_step_match = PRETTY_STEP_RE.search(raw_line)
        if pretty_step_match:
            step = int(pretty_step_match.group(1))
            row = by_step.setdefault(step, {"step": step})
            for src_key, dst_key in (
                ("reward", "critic/rewards/mean"),
                ("resp_len", "response_length/mean"),
                ("entropy", "actor/entropy"),
                ("val", "val-core/alphaagentevo/reward/mean@3"),
            ):
                value = extract_pretty_float(raw_line, src_key)
                if value is not None:
                    row[dst_key] = value
            continue

        pretty_val_match = PRETTY_VAL_RE.search(raw_line)
        if pretty_val_match:
            step = int(pretty_val_match.group(1))
            row = by_step.setdefault(step, {"step": step})
            for src_key, dst_key in (
                ("reward", "val-core/alphaagentevo/reward/mean@3"),
                ("vr", "val-paper/vr"),
                ("pass@3", "val-paper/pass@3"),
                ("pass@5", "val-paper/pass@5"),
                ("beat_rate", "val-paper/beat_rate"),
                ("best_metric_mean", "val-paper/best_metric_mean"),
            ):
                value = extract_pretty_float(raw_line, src_key)
                if value is not None:
                    row[dst_key] = value
            continue

        if "step:" not in raw_line or "critic/rewards/mean:" not in raw_line:
            continue

        step_match = STEP_RE.search(raw_line)
        if not step_match:
            continue

        step = int(step_match.group(1))
        row = by_step.setdefault(step, {"step": step})

        for key in (
            "critic/rewards/mean",
            "response_length/mean",
            "actor/entropy",
            "val-core/alphaagentevo/reward/mean@3",
        ):
            value = extract_float(raw_line, key)
            if value is not None:
                row[key] = value

        for metric_name in (
            "vr",
            "pass@3",
            "pass@5",
            "beat_rate",
            "best_metric_mean",
        ):
            value = extract_val_paper_metric(raw_line, metric_name)
            if value is not None:
                row[f"val-paper/{metric_name}"] = value

    return [by_step[step] for step in sorted(by_step)]


def plot_metrics(rows: list[dict[str, float]], output_path: Path) -> None:
    if not rows:
        raise RuntimeError("No step metrics found in log.")

    steps = [int(row["step"]) for row in rows]
    rewards = [row.get("critic/rewards/mean") for row in rows]
    response_lengths = [row.get("response_length/mean") for row in rows]
    entropies = [row.get("actor/entropy") for row in rows]

    val_steps = [int(row["step"]) for row in rows if "val-core/alphaagentevo/reward/mean@3" in row]
    val_rewards = [row["val-core/alphaagentevo/reward/mean@3"] for row in rows if "val-core/alphaagentevo/reward/mean@3" in row]
    paper_val_steps = [int(row["step"]) for row in rows if any(key.startswith("val-paper/") for key in row)]
    vr_values = [row["val-paper/vr"] for row in rows if "val-paper/vr" in row]
    pass3_values = [row["val-paper/pass@3"] for row in rows if "val-paper/pass@3" in row]
    pass5_values = [row["val-paper/pass@5"] for row in rows if "val-paper/pass@5" in row]
    beat_rate_values = [row["val-paper/beat_rate"] for row in rows if "val-paper/beat_rate" in row]
    best_metric_values = [row["val-paper/best_metric_mean"] for row in rows if "val-paper/best_metric_mean" in row]

    has_paper_metrics = any(
        key in row
        for row in rows
        for key in (
            "val-paper/vr",
            "val-paper/pass@3",
            "val-paper/pass@5",
            "val-paper/beat_rate",
            "val-paper/best_metric_mean",
        )
    )
    fig = plt.figure(figsize=(16, 8 if has_paper_metrics else 4))
    nrows = 2 if has_paper_metrics else 1

    ax1 = fig.add_subplot(nrows, 3, 1)
    ax1.plot(steps, rewards, marker="o", label="train reward")
    if val_steps:
        if len(val_steps) > 1:
            ax1.plot(val_steps, val_rewards, color="tab:orange", alpha=0.5, linewidth=1)
        ax1.scatter(val_steps, val_rewards, marker="s", label="val reward", color="tab:orange", zorder=3)
        for x, y in zip(val_steps, val_rewards):
            ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax1.set_xlabel("step")
    ax1.set_ylabel("reward")
    ax1.set_title("Training reward vs validation reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(nrows, 3, 2)
    ax2.plot(steps, rewards, marker="o", label="train reward")
    ax2.set_xlabel("step")
    ax2.set_ylabel("training reward")
    ax2.set_title("Training reward and response length")
    ax2.grid(True, alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(steps, response_lengths, color="green", marker="s", label="response len")
    ax2b.set_ylabel("response length")

    ax3 = fig.add_subplot(nrows, 3, 3)
    ax3.plot(steps, rewards, marker="o", label="train reward")
    ax3.set_xlabel("step")
    ax3.set_ylabel("training reward")
    ax3.set_title("Training reward and entropy")
    ax3.grid(True, alpha=0.3)
    ax3b = ax3.twinx()
    ax3b.plot(steps, entropies, color="gold", marker="s", label="entropy")
    ax3b.set_ylabel("entropy")

    if has_paper_metrics:
        ax4 = fig.add_subplot(2, 3, 4)
        if paper_val_steps and vr_values:
            ax4.plot(
                [int(row["step"]) for row in rows if "val-paper/vr" in row],
                vr_values,
                marker="o",
                label="VR",
            )
        ax4.set_xlabel("step")
        ax4.set_ylabel("valid ratio")
        ax4.set_title("Validation VR")
        ax4.grid(True, alpha=0.3)
        if vr_values:
            ax4.legend()

        ax5 = fig.add_subplot(2, 3, 5)
        if pass3_values:
            ax5.plot(
                [int(row["step"]) for row in rows if "val-paper/pass@3" in row],
                pass3_values,
                marker="o",
                label="pass@3",
            )
        if pass5_values:
            ax5.plot(
                [int(row["step"]) for row in rows if "val-paper/pass@5" in row],
                pass5_values,
                marker="s",
                label="pass@5",
            )
        if beat_rate_values:
            ax5.plot(
                [int(row["step"]) for row in rows if "val-paper/beat_rate" in row],
                beat_rate_values,
                marker="^",
                label="beat_rate",
            )
        ax5.set_xlabel("step")
        ax5.set_ylabel("rate")
        ax5.set_title("Validation Pass/Beat Rate")
        ax5.grid(True, alpha=0.3)
        if pass3_values or pass5_values or beat_rate_values:
            ax5.legend()

        ax6 = fig.add_subplot(2, 3, 6)
        if best_metric_values:
            ax6.plot(
                [int(row["step"]) for row in rows if "val-paper/best_metric_mean" in row],
                best_metric_values,
                marker="o",
                color="crimson",
                label="best_metric_mean",
            )
        ax6.set_xlabel("step")
        ax6.set_ylabel("metric")
        ax6.set_title("Validation Best Metric")
        ax6.grid(True, alpha=0.3)
        if best_metric_values:
            ax6.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="/kaggle/working/aae_v2/logs/train.pretty.log",
        help="Path to train.pretty.log or train.live.log",
    )
    parser.add_argument(
        "--out",
        default="/kaggle/working/aae_v2/plots/train_metrics.png",
        help="Output image path",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists() and log_path.name == "train.pretty.log":
        fallback = log_path.with_name("train.live.log")
        if fallback.exists():
            log_path = fallback
    output_path = Path(args.out)
    rows = parse_step_metrics(log_path.read_text(errors="replace"))
    plot_metrics(rows, output_path)

    print(f"parsed_steps={[int(row['step']) for row in rows]}")
    print(f"validation_steps={[int(row['step']) for row in rows if 'val-core/alphaagentevo/reward/mean@3' in row]}")
    print(
        "validation_points="
        + str(
            [
                (int(row["step"]), float(row["val-core/alphaagentevo/reward/mean@3"]))
                for row in rows
                if "val-core/alphaagentevo/reward/mean@3" in row
            ]
        )
    )
    print(f"paper_validation_steps={[int(row['step']) for row in rows if any(key.startswith('val-paper/') for key in row)]}")
    print(f"saved_plot={output_path}")


if __name__ == "__main__":
    main()
