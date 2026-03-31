#!/usr/bin/env python3
"""Fail-fast runtime compatibility checks for AlphaAgentEvo v2."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_VERSIONS = {
    "torch": "2.6.0",
    "tensordict": "0.6.2",
    "sglang": "0.4.6.post5",
    "transformers": "4.51.1",
    "tokenizers": "0.21.4",
    "ray": "2.54.0",
    "verl": "0.4.1",
}


def _collect_version_errors() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    errors: list[str] = []

    for pkg, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(pkg)
            versions[pkg] = actual
            if actual != expected:
                errors.append(f"{pkg}: expected {expected}, got {actual}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{pkg}: not installed")
    return versions, errors


def _check_patch_markers(work_dir: Path) -> list[str]:
    errors: list[str] = []
    markers = [
        (
            work_dir / "verl/verl/tools/factor_tool.py",
            "def rollout_trace_op(func):",
            "factor_tool rollout_trace fallback",
        ),
        (
            work_dir / "verl/verl/workers/actor/dp_actor.py",
            "_FLASH_ATTN_BERT_PADDING_AVAILABLE = False",
            "dp_actor flash_attn guard",
        ),
        (
            work_dir / "verl/verl/workers/critic/dp_critic.py",
            "_FLASH_ATTN_BERT_PADDING_AVAILABLE = False",
            "dp_critic flash_attn guard",
        ),
    ]

    for path, marker, name in markers:
        if not path.is_file():
            errors.append(f"{name}: missing file {path}")
            continue
        text = path.read_text()
        if marker not in text:
            errors.append(f"{name}: marker not found in {path}")
    return errors


def _run_pip_check() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return []
    return ["pip check failed:\n" + result.stdout.strip()]


def _check_imports() -> list[str]:
    errors: list[str] = []
    try:
        import torch  # noqa: F401
        import sglang  # noqa: F401
        import tensordict  # noqa: F401
        import verl  # noqa: F401
        from verl.tools.factor_tool import FactorTool  # noqa: F401
    except Exception as exc:
        errors.append(f"import check failed: {exc!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--skip-pip-check", action="store_true")
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 10):
        print(f"ERROR: Python 3.10 required, got {sys.version.split()[0]}")
        return 1

    work_dir = Path(args.work_dir).resolve()
    if not work_dir.exists():
        print(f"ERROR: work dir does not exist: {work_dir}")
        return 1

    versions, errors = _collect_version_errors()
    errors.extend(_check_patch_markers(work_dir))
    errors.extend(_check_imports())
    if not args.skip_pip_check:
        errors.extend(_run_pip_check())

    print("Runtime versions:")
    print(json.dumps(versions, indent=2, sort_keys=True))

    if errors:
        print("\nPreflight failed with issues:")
        for issue in errors:
            print(f"- {issue}")
        return 1

    print("\nRuntime preflight: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
