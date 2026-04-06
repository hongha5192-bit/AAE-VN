"""
Phase 2: Evaluate best evolved factors through the full Qlib LightGBM pipeline.

Steps:
1. Load best evolved factors from evo_results/
2. Compute raw factor values using the AAE-VN expression engine
3. Create combined_factors_df.pkl
4. Copy run22 best workspace config + run qrun
5. Compare IR against run22's baseline (0.9282)

Usage:
    python phase2_qlib_eval.py [--top N] [--workspace_name myrun]
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVO_RESULTS_DIR = SCRIPT_DIR / "evo_results"
RUN22_BEST_WORKSPACE = PROJECT_ROOT / "workspaces" / "run22" / "a55d3023244645c2aa5025bee761a1ca"
WORKSPACES_DIR = PROJECT_ROOT / "workspaces"

# Add AAE-VN to path for backtest module
sys.path.insert(0, str(SCRIPT_DIR))

# ── Run22 baseline ────────────────────────────────────────────────────────────
RUN22_IR = 0.9282
RUN22_AR = 0.2326


def load_best_evolved_factors(top_n: int = 5) -> list[dict]:
    """Load best evolved factors from evo_results summaries."""
    summaries = sorted(EVO_RESULTS_DIR.glob("summary_*.json"), reverse=True)
    all_factors = sorted(EVO_RESULTS_DIR.glob("all_factors_*.json"), reverse=True)

    best_factors = []
    seen_exprs = set()

    for summary_file in summaries:
        with open(summary_file) as f:
            summary = json.load(f)
        for s in summary:
            expr = s.get("best_expr", "")
            ir = s.get("best_Information_Ratio_with_cost", s.get("best_IC", -np.inf))
            if not expr or expr in seen_exprs:
                continue
            seen_exprs.add(expr)
            best_factors.append({
                "name": f"evo_{s['seed_name']}",
                "expr": expr,
                "IR_proxy": float(ir) if ir is not None else -np.inf,
                "seed_name": s["seed_name"],
            })

    # Also pull from all_factors files
    for all_file in all_factors:
        with open(all_file) as f:
            all_f = json.load(f)
        for f_entry in all_f:
            expr = f_entry.get("expr", "")
            ir_key = next((k for k in f_entry if "Information_Ratio" in k or k == "IC"), None)
            ir = f_entry.get(ir_key, -np.inf) if ir_key else -np.inf
            if not expr or expr in seen_exprs or ir is None:
                continue
            seen_exprs.add(expr)
            best_factors.append({
                "name": f"evo_{f_entry.get('name', 'factor')}",
                "expr": expr,
                "IR_proxy": float(ir) if ir is not None else -np.inf,
                "seed_name": f_entry.get("name", ""),
            })

    # Sort by IR_proxy descending, take top_n
    best_factors.sort(key=lambda x: x["IR_proxy"], reverse=True)
    valid = [f for f in best_factors if not np.isnan(f["IR_proxy"]) and f["IR_proxy"] > 0]

    print(f"\nFound {len(valid)} valid evolved factors (IR > 0)")
    for i, f in enumerate(valid[:top_n]):
        print(f"  {i+1}. {f['name']}: IR_proxy={f['IR_proxy']:.4f}")
        print(f"     expr: {f['expr'][:80]}...")

    return valid[:top_n]


def compute_factor_values(factor_name: str, factor_expr: str) -> pd.Series | None:
    """Compute raw factor values using AAE-VN expression engine."""
    from backtest.factor_executor import load_data, _cached_columns
    from expression_manager.expr_parser import parse_expression, parse_symbol
    from expression_manager import function_lib
    import re

    df = load_data()

    _PYTHON_KEYWORDS = {'return', 'class', 'import', 'from', 'def', 'if', 'else',
                        'for', 'while', 'try', 'except', 'with', 'as', 'in', 'is',
                        'not', 'and', 'or', 'pass', 'break', 'continue', 'yield',
                        'lambda', 'global', 'nonlocal', 'del', 'raise', 'assert'}

    try:
        # Parse expression
        parsed_code = parse_expression(factor_expr)

        # Build safe name mapping
        columns = list(df.columns)
        safe_name_map = {}
        for col in columns:
            clean = col.replace('$', '')
            safe_name_map[col] = f'col_{clean}' if clean in _PYTHON_KEYWORDS else clean

        executable_code = parse_symbol(parsed_code, columns)
        for col in columns:
            clean = col.replace('$', '')
            safe = safe_name_map[col]
            if clean != safe:
                executable_code = re.sub(r'\b' + re.escape(clean) + r'\b', safe, executable_code)

        # Build namespace
        exec_namespace = {}
        for name in dir(function_lib):
            obj = getattr(function_lib, name)
            if callable(obj) and not name.startswith('_'):
                exec_namespace[name] = obj
        exec_namespace['np'] = np
        exec_namespace['pd'] = pd
        for col in columns:
            exec_namespace[safe_name_map[col]] = _cached_columns[col].copy()

        # Execute
        factor_values = eval(executable_code, exec_namespace)

        if isinstance(factor_values, pd.DataFrame):
            factor_values = factor_values.iloc[:, 0]
        elif isinstance(factor_values, np.ndarray):
            factor_values = pd.Series(factor_values, index=df.index)

        print(f"  Computed {factor_name}: {factor_values.notna().sum()} non-nan values")
        return factor_values

    except Exception as e:
        print(f"  ERROR computing {factor_name}: {e}")
        return None


def build_combined_pkl(factors: list[dict], output_path: Path) -> bool:
    """Build combined_factors_df.pkl from evolved factor expressions."""
    print(f"\nComputing factor values for {len(factors)} factors...")

    factor_series = {}
    for f in factors:
        vals = compute_factor_values(f["name"], f["expr"])
        if vals is not None:
            # Sanitize name for use as column
            safe_name = f["name"].replace("-", "_").replace(" ", "_")[:50]
            factor_series[safe_name] = vals

    if not factor_series:
        print("ERROR: No factors computed successfully")
        return False

    print(f"\nBuilding DataFrame with {len(factor_series)} factors...")

    # Build DataFrame with MultiIndex columns like run22
    dfs = []
    for name, series in factor_series.items():
        col_df = series.to_frame(name=('feature', name))
        col_df.columns = pd.MultiIndex.from_tuples([('feature', name)])
        dfs.append(col_df)

    combined = pd.concat(dfs, axis=1)
    combined.index.names = ['datetime', 'instrument']

    print(f"Combined DataFrame shape: {combined.shape}")
    print(f"Date range: {combined.index.get_level_values('datetime').min()} to "
          f"{combined.index.get_level_values('datetime').max()}")

    with open(output_path, 'wb') as f:
        pickle.dump(combined, f)

    print(f"Saved: {output_path}")
    return True


def create_workspace(workspace_name: str, factors: list[dict]) -> Path:
    """Create new workspace with evolved factors."""
    workspace_id = workspace_name or str(uuid.uuid4()).replace('-', '')[:32]
    workspace_dir = WORKSPACES_DIR / f"evo_{workspace_id}"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Build combined_factors_df.pkl
    pkl_path = workspace_dir / "combined_factors_df.pkl"
    if not build_combined_pkl(factors, pkl_path):
        return None

    # Copy config from run22 best workspace
    src_config = RUN22_BEST_WORKSPACE / "conf_vn_combined_kdd_ver.yaml"
    dst_config = workspace_dir / "conf_vn_combined_kdd_ver.yaml"
    shutil.copy(src_config, dst_config)

    # Copy read_exp_res.py
    shutil.copy(RUN22_BEST_WORKSPACE / "read_exp_res.py", workspace_dir / "read_exp_res.py")

    # Save factor info
    with open(workspace_dir / "factors.json", "w") as f:
        json.dump(factors, f, indent=2)

    print(f"\nWorkspace created: {workspace_dir}")
    return workspace_dir


def run_qlib_pipeline(workspace_dir: Path) -> dict | None:
    """Run Qlib pipeline and return metrics."""
    config_file = workspace_dir / "conf_vn_combined_kdd_ver.yaml"

    print(f"\nRunning Qlib pipeline in {workspace_dir}...")
    print("(This may take 10-20 minutes)")

    result = subprocess.run(
        ["qrun", str(config_file)],
        capture_output=True,
        text=True,
        cwd=str(workspace_dir),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )

    if result.returncode != 0:
        print("ERROR running qrun:")
        print(result.stderr[-2000:])
        return None

    print("qrun completed successfully")

    # Parse results
    try:
        result_py = subprocess.run(
            ["python", "read_exp_res.py"],
            capture_output=True,
            text=True,
            cwd=str(workspace_dir),
        )
        if result_py.returncode == 0:
            metrics_df = pd.read_csv(workspace_dir / "qlib_res.csv", index_col=0)
            metrics = metrics_df[metrics_df.columns[0]].to_dict()
            return metrics
    except Exception as e:
        print(f"Error reading results: {e}")

    return None


def print_comparison(metrics: dict, factors: list[dict]):
    """Print comparison against run22 baseline."""
    ir = metrics.get("1day.excess_return_with_cost.information_ratio", np.nan)
    ar = metrics.get("1day.excess_return_with_cost.annualized_return", np.nan)
    mdd = metrics.get("1day.excess_return_with_cost.max_drawdown", np.nan)

    print(f"\n{'='*60}")
    print(f"PHASE 2 RESULTS — Evolved Factors vs Run22 Baseline")
    print(f"{'='*60}")
    print(f"Factors used: {[f['name'] for f in factors]}")
    print(f"\n{'Metric':<40} {'Run22':>10} {'Evolved':>10} {'Delta':>10}")
    print(f"{'-'*70}")
    print(f"{'IR with cost':<40} {RUN22_IR:>10.4f} {ir:>10.4f} {ir-RUN22_IR:>+10.4f}")
    print(f"{'AR with cost':<40} {RUN22_AR:>10.4f} {ar:>10.4f} {ar-RUN22_AR:>+10.4f}")
    print(f"{'MDD':<40} {'-0.4314':>10} {mdd:>10.4f}")

    if ir > RUN22_IR:
        print(f"\n*** SUCCESS: Evolved factors BEAT run22 by {ir-RUN22_IR:+.4f} IR ***")
    else:
        print(f"\nResult: Evolved factors below run22 by {RUN22_IR-ir:.4f} IR")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=3, help="Top N evolved factors to use")
    parser.add_argument("--workspace_name", type=str, default=None, help="Workspace name")
    parser.add_argument("--dry_run", action="store_true", help="Only compute factors, skip qrun")
    parser.add_argument("--factors_file", type=str, default=None,
                        help="JSON file with factor list: [{name, expr, IR_proxy}]")
    args = parser.parse_args()

    # Load evolved factors
    if args.factors_file:
        with open(args.factors_file) as f:
            factors = json.load(f)
        factors = factors[:args.top]
    else:
        factors = load_best_evolved_factors(top_n=args.top)

    if not factors:
        print("No evolved factors found. Run evolution first.")
        return

    # Create workspace
    workspace_dir = create_workspace(args.workspace_name, factors)
    if workspace_dir is None:
        print("Failed to create workspace")
        return

    if args.dry_run:
        print("\nDry run complete. Workspace ready at:", workspace_dir)
        return

    # Run pipeline
    metrics = run_qlib_pipeline(workspace_dir)
    if metrics is None:
        print("Pipeline failed")
        return

    print_comparison(metrics, factors)


if __name__ == "__main__":
    main()
