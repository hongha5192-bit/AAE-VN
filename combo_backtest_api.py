"""
Combo Backtest API — LGBModel multi-factor evaluator.

Bypasses Qlib entirely. Uses daily_pv.h5 + expression_manager + LightGBM.

Replicates AlphaAgent run22 pipeline:
  1. Compute factor expressions → feature matrix
  2. Compute 4 base features (same as Qlib conf)
  3. Train LightGBM (2016-2020), validate (2021)
  4. Predict on test (2022-2025)
  5. Build TopkDropout portfolio
  6. Return IR, AR, MDD

Usage:
    python combo_backtest_api.py                     # start API on port 8003
    python combo_backtest_api.py --test               # run baseline test
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Add AAE-VN to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from expression_manager.expr_parser import parse_expression, parse_symbol
from expression_manager import function_lib

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH = SCRIPT_DIR / "backtest" / "data" / "daily_pv.h5"

# LightGBM params from run22 conf_vn_combined_kdd_ver.yaml
LGB_PARAMS = {
    "objective": "mse",
    "colsample_bytree": 0.8879,
    "learning_rate": 0.1,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 4,
    "num_leaves": 210,
    "num_threads": 20,
    "verbosity": -1,
}

# Periods matching run22
TRAIN_START, TRAIN_END = "2016-01-01", "2020-12-31"
VALID_START, VALID_END = "2021-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2025-12-31"

# Portfolio params matching run22
TOPK = 10
N_DROP = 2
HOLD_THRESH = 2
REBALANCE_FREQ = 5
COST_BUY = 0.0013
COST_SELL = 0.0013
ANN_SCALER = 252

# Python keywords that need safe naming
_PYTHON_KEYWORDS = {'return', 'class', 'import', 'from', 'def', 'if', 'else',
                    'for', 'while', 'try', 'except', 'with', 'as', 'in', 'is',
                    'not', 'and', 'or', 'pass', 'break', 'continue', 'yield',
                    'lambda', 'global', 'nonlocal', 'del', 'raise', 'assert'}

# ── Cached data ────────────────────────────────────────────────────────────
_df: pd.DataFrame = None
_cached_columns: dict = {}


def load_data():
    """Load daily_pv.h5 into memory."""
    global _df, _cached_columns
    if _df is not None:
        return _df

    print(f"[Combo] Loading {DATA_PATH}...")
    _df = pd.read_hdf(str(DATA_PATH))
    for col in _df.columns:
        _cached_columns[col] = _df[col]

    print(f"[Combo] Loaded {_df.shape[0]} rows, {len(_df.columns)} cols, "
          f"{_df.index.get_level_values('instrument').nunique()} instruments")
    return _df


def compute_expression(expr_str: str) -> pd.Series:
    """Compute a factor expression and return Series aligned with _df index."""
    import re

    df = load_data()
    parsed_code = parse_expression(expr_str)
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

    exec_namespace = {}
    for name in dir(function_lib):
        obj = getattr(function_lib, name)
        if callable(obj) and not name.startswith('_'):
            exec_namespace[name] = obj
    exec_namespace['np'] = np
    exec_namespace['pd'] = pd
    for col in columns:
        exec_namespace[safe_name_map[col]] = _cached_columns[col].copy()

    result = eval(executable_code, exec_namespace)

    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    elif isinstance(result, np.ndarray):
        result = pd.Series(result, index=df.index)

    return result


def compute_base_features() -> pd.DataFrame:
    """Compute 4 base features matching run22's QlibDataLoader config."""
    df = load_data()

    features = {}

    # Feature 1: ($close - $open) / $open
    features["base_co_ratio"] = (_cached_columns["$close"] - _cached_columns["$open"]) / (_cached_columns["$open"] + 1e-12)

    # Feature 2: $close / Ref($close, 1) - 1  (daily return)
    features["base_return"] = _cached_columns["$close"].groupby("instrument").transform(
        lambda x: x / x.shift(1) - 1
    )

    # Feature 3: $volume / Mean($volume, 20)
    features["base_vol_ratio"] = _cached_columns["$volume"] / _cached_columns["$volume"].groupby("instrument").transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )

    # Feature 4: ($high - $low) / Ref($close, 1)
    prev_close = _cached_columns["$close"].groupby("instrument").transform(lambda x: x.shift(1))
    features["base_hl_ratio"] = (_cached_columns["$high"] - _cached_columns["$low"]) / (prev_close + 1e-12)

    return pd.DataFrame(features, index=df.index)


def compute_label() -> pd.Series:
    """Compute label: Ref($close, -2) / Ref($close, -1) - 1 (next-day return)."""
    close = _cached_columns["$close"]
    # Ref($close, -2) = close shifted 2 days forward, Ref($close, -1) = close shifted 1 day forward
    fwd1 = close.groupby("instrument").transform(lambda x: x.shift(-1))
    fwd2 = close.groupby("instrument").transform(lambda x: x.shift(-2))
    label = fwd2 / (fwd1 + 1e-12) - 1
    return label


def cs_zscore_norm(series: pd.Series) -> pd.Series:
    """Cross-sectional Z-score normalization (per date)."""
    return series.groupby("datetime").transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))


def build_dataset(factor_exprs: Dict[str, str], fixed_factors_pkl: Optional[str] = None):
    """Build feature matrix + label, split into train/valid/test.

    Args:
        factor_exprs: {name: expression} for new factors to compute
        fixed_factors_pkl: path to pkl with pre-computed fixed factors (optional)
    """
    df = load_data()
    dates = df.index.get_level_values("datetime")

    # 1. Compute base features
    print("[Combo] Computing base features...")
    base_features = compute_base_features()

    # 2. Compute new factor expressions
    new_factors = {}
    for name, expr in factor_exprs.items():
        print(f"[Combo] Computing factor: {name}")
        try:
            new_factors[name] = compute_expression(expr)
        except Exception as e:
            print(f"[Combo] ERROR computing {name}: {e}")
            return None

    new_factors_df = pd.DataFrame(new_factors, index=df.index) if new_factors else pd.DataFrame(index=df.index)

    # 3. Load fixed factors from pkl
    if fixed_factors_pkl and Path(fixed_factors_pkl).exists():
        print(f"[Combo] Loading fixed factors from {fixed_factors_pkl}")
        with open(fixed_factors_pkl, "rb") as f:
            fixed_df = pickle.load(f)
        # Flatten MultiIndex columns if needed
        if isinstance(fixed_df.columns, pd.MultiIndex):
            fixed_df.columns = [c[1] if isinstance(c, tuple) else c for c in fixed_df.columns]
        # Align index
        fixed_df = fixed_df.reindex(df.index)
    else:
        fixed_df = pd.DataFrame(index=df.index)

    # 4. Combine all features
    all_features = pd.concat([base_features, fixed_df, new_factors_df], axis=1)

    # 5. Compute label
    print("[Combo] Computing label...")
    label = compute_label()

    # 6. Fillna + CSZScoreNorm (matching Qlib processors)
    print("[Combo] Normalizing features...")
    all_features = all_features.fillna(0)

    for col in all_features.columns:
        all_features[col] = cs_zscore_norm(all_features[col])

    label = label.dropna()
    label = cs_zscore_norm(label)

    # 7. Align features and label
    common_idx = all_features.index.intersection(label.index)
    all_features = all_features.loc[common_idx]
    label = label.loc[common_idx]

    # 8. Split by date
    split_dates = all_features.index.get_level_values("datetime")

    train_mask = (split_dates >= TRAIN_START) & (split_dates <= TRAIN_END)
    valid_mask = (split_dates >= VALID_START) & (split_dates <= VALID_END)
    test_mask = (split_dates >= TEST_START) & (split_dates <= TEST_END)

    result = {
        "X_train": all_features[train_mask], "y_train": label[train_mask],
        "X_valid": all_features[valid_mask], "y_valid": label[valid_mask],
        "X_test": all_features[test_mask], "y_test": label[test_mask],
        "feature_names": list(all_features.columns),
    }

    print(f"[Combo] Dataset: train={train_mask.sum()}, valid={valid_mask.sum()}, "
          f"test={test_mask.sum()}, features={len(result['feature_names'])}")

    return result


def train_lgb(data: dict) -> lgb.Booster:
    """Train LightGBM model."""
    print("[Combo] Training LightGBM...")
    t0 = time.time()

    train_set = lgb.Dataset(data["X_train"], label=data["y_train"])
    valid_set = lgb.Dataset(data["X_valid"], label=data["y_valid"])

    model = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    elapsed = time.time() - t0
    print(f"[Combo] LGBModel trained in {elapsed:.1f}s, {model.best_iteration} rounds")
    return model


def backtest_portfolio(predictions: pd.Series, df: pd.DataFrame) -> dict:
    """Signal-based portfolio backtest using vectorized operations.

    Matches Qlib's approach:
    - Every REBALANCE_FREQ days, select top TOPK stocks by prediction score
    - Equal-weight portfolio
    - Daily return = mean return of held stocks
    - Excess return = portfolio return - benchmark return
    - Transaction cost applied on turnover

    Args:
        predictions: predicted scores indexed by (datetime, instrument)
        df: full dataframe with $close, $bench_return columns
    """
    # Pivot predictions and prices to (date × instrument) matrices
    pred_df = predictions.unstack("instrument")
    close_df = df["$close"].unstack("instrument")

    # Benchmark daily return
    if "$bench_return" in df.columns:
        bench_daily = df["$bench_return"].groupby("datetime").first()
    else:
        bench_daily = pd.Series(0.0, index=close_df.index)

    # Daily stock returns: close-to-close (matching Qlib deal_price: close)
    stock_returns = close_df.pct_change()

    # Filter to test period
    test_dates = sorted(pred_df.index)

    # Build holding matrix using TopkDropout strategy
    # At each rebalance: keep stocks still in top (topk+n_drop), drop worst, buy best new
    holdings = pd.DataFrame(0.0, index=test_dates, columns=pred_df.columns)

    current_held = []  # ordered list of held instruments
    hold_days = {}  # {instrument: days_held}

    for i, date in enumerate(test_dates):
        # Track hold days
        for inst in current_held:
            hold_days[inst] = hold_days.get(inst, 0) + 1

        if i % REBALANCE_FREQ == 0:
            day_pred = pred_df.loc[date].dropna()
            if len(day_pred) >= TOPK:
                ranked = day_pred.sort_values(ascending=False)
                top_candidates = set(ranked.head(TOPK + N_DROP).index)

                # Keep: held stocks still in top (topk + n_drop)
                keep = [s for s in current_held if s in top_candidates]

                # If too many keepers, drop worst-ranked among them
                if len(keep) > TOPK - N_DROP:
                    keep_ranked = ranked.reindex(keep).dropna().sort_values(ascending=False)
                    keep = keep_ranked.head(TOPK - N_DROP).index.tolist()

                # Fill remaining slots with best non-held stocks
                n_buy = TOPK - len(keep)
                buy = [s for s in ranked.index if s not in keep][:n_buy]

                new_held = (keep + buy)[:TOPK]
                for s in buy:
                    hold_days[s] = 0
                for s in current_held:
                    if s not in new_held:
                        hold_days.pop(s, None)
                current_held = new_held

        # Set equal-weight holdings
        weight = 1.0 / max(len(current_held), 1)
        for inst in current_held:
            if inst in holdings.columns:
                holdings.loc[date, inst] = weight

    # Features at day t-1 → select stocks → open position at open of day t
    # holdings[t-1] marks the selection day → position earns return on day t (open_t to open_{t+1})
    # So shift holdings by 1: signal at t-1, returns earned at t
    holdings = holdings.shift(1).fillna(0)

    # Daily portfolio return (equal-weight)
    port_ret = (holdings * stock_returns.reindex(index=holdings.index, columns=holdings.columns).fillna(0)).sum(axis=1)

    # Align benchmark
    bench_aligned = bench_daily.reindex(port_ret.index).fillna(0)

    # Transaction cost: estimate turnover at each rebalance
    turnover = holdings.diff().abs().sum(axis=1)
    cost = turnover * (COST_BUY + COST_SELL) / 2

    # Excess returns
    excess_woc = port_ret - bench_aligned
    excess_wc = excess_woc - cost

    # Drop NaN
    excess_wc = excess_wc.replace([np.inf, -np.inf], 0).fillna(0)
    excess_woc = excess_woc.replace([np.inf, -np.inf], 0).fillna(0)

    # IR = annualized Sharpe of excess returns
    ir_wc = (excess_wc.mean() / (excess_wc.std() + 1e-12)) * np.sqrt(ANN_SCALER)
    ir_woc = (excess_woc.mean() / (excess_woc.std() + 1e-12)) * np.sqrt(ANN_SCALER)

    # Annualized return
    ar_wc = excess_wc.mean() * ANN_SCALER
    ar_woc = excess_woc.mean() * ANN_SCALER

    # Max drawdown
    cum_excess = (1 + excess_wc).cumprod()
    running_max = cum_excess.cummax()
    drawdown = (cum_excess - running_max) / running_max
    mdd_wc = float(drawdown.min())

    return {
        "IR": float(np.round(ir_wc, 4)),
        "IR_without_cost": float(np.round(ir_woc, 4)),
        "AR": float(np.round(ar_wc, 4)),
        "AR_without_cost": float(np.round(ar_woc, 4)),
        "MDD": float(np.round(mdd_wc, 4)),
        "n_days": len(excess_wc),
        "mean_daily_excess": float(np.round(excess_wc.mean(), 6)),
        "success": True,
    }


def evaluate_combo(
    factor_exprs: Dict[str, str],
    fixed_factors_pkl: Optional[str] = None,
) -> dict:
    """Full combo evaluation: compute → train → predict → backtest."""
    t0 = time.time()

    # Build dataset
    data = build_dataset(factor_exprs, fixed_factors_pkl)
    if data is None:
        return {"success": False, "error": "Failed to build dataset"}

    # Train
    model = train_lgb(data)

    # Predict on test set
    print("[Combo] Predicting on test set...")
    test_preds = model.predict(data["X_test"])
    pred_series = pd.Series(test_preds, index=data["X_test"].index, name="prediction")

    # IC on test set
    ic = pred_series.groupby("datetime").apply(
        lambda x: x.corr(data["y_test"].loc[x.index]) if len(x) > 2 else np.nan
    )
    ic_mean = float(ic.mean())
    icir = float(ic.mean() / (ic.std() + 1e-12))

    # Rank IC
    rank_ic = pred_series.groupby("datetime").apply(
        lambda x: x.rank().corr(data["y_test"].loc[x.index].rank()) if len(x) > 2 else np.nan
    )
    rank_ic_mean = float(rank_ic.mean())

    # Portfolio backtest
    print("[Combo] Running portfolio backtest...")
    df = load_data()
    port_metrics = backtest_portfolio(pred_series, df)

    elapsed = time.time() - t0

    result = {
        **port_metrics,
        "IC": float(np.round(ic_mean, 6)),
        "ICIR": float(np.round(icir, 4)),
        "RankIC": float(np.round(rank_ic_mean, 6)),
        "features": data["feature_names"],
        "n_features": len(data["feature_names"]),
        "lgb_rounds": model.best_iteration,
        "elapsed": float(np.round(elapsed, 1)),
    }

    print(f"[Combo] Done in {elapsed:.1f}s — IR={result['IR']}, AR={result['AR']}, MDD={result['MDD']}")
    return result


# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title="Combo Backtest API", version="1.0")


class ComboRequest(BaseModel):
    factor_exprs: Dict[str, str] = Field(
        ..., description="New factor expressions: {name: expr}"
    )
    fixed_factors_pkl: Optional[str] = Field(
        None, description="Path to pkl with pre-computed fixed factors"
    )


@app.on_event("startup")
async def startup():
    load_data()
    # Pre-compute base features
    compute_base_features()
    print("[Combo] API ready on /combo_backtest")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Combo Backtest API"}


@app.post("/combo_backtest")
async def combo_backtest(req: ComboRequest):
    try:
        result = evaluate_combo(
            factor_exprs=req.factor_exprs,
            fixed_factors_pkl=req.fixed_factors_pkl,
        )
        return {"data": result}
    except Exception as e:
        import traceback
        return {"data": None, "error": str(e), "traceback": traceback.format_exc()}


# ── CLI test mode ──────────────────────────────────────────────────────────
def test_baseline():
    """Test with run22 Loop 14's 3 factors to verify IR matches."""
    print("=" * 60)
    print("BASELINE TEST: Run22 Loop 14 (3 factors)")
    print("Expected: IR ≈ +0.928, AR ≈ +23.3%")
    print("=" * 60)

    factor_exprs = {
        "VolScaled5D_ZScoreMom_FlowSurprise":
            "ZSCORE(TS_ZSCORE(TS_SUM($return, 5), 5) * (($net_foreign_val - TS_MEDIAN($net_foreign_val, 20)) / (TS_MEAN(MAX(MAX($high - $low, ABS($high - $close)), ABS($low - $close)), 14) + 1e-8)))",
        "EMA5_10_Crossover_LogFlow_PoweredMom":
            "RANK((EMA($close, 5)/EMA($close, 10) - 1) * TS_ZSCORE(LOG(1 + ($net_foreign_val - TS_MEDIAN($net_foreign_val, 20)) / (TS_STD($net_foreign_val, 20) + 1e-8)), 10))",
        "DecayFlowStdMom_Z":
            "ZSCORE(TS_RANK(TS_PCTCHANGE($close, 5), 5) * (DECAYLINEAR($net_foreign_val - DELAY($net_foreign_val, 1), 10) / (TS_STD($return, 14) + 1e-8)))",
    }

    result = evaluate_combo(factor_exprs)
    print()
    print("=" * 60)
    print("RESULTS:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run baseline test")
    parser.add_argument("--port", type=int, default=8003, help="API port")
    args = parser.parse_args()

    if args.test:
        test_baseline()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)
