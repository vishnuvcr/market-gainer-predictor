import argparse
import glob
import os
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "rs_nifty_1d",
    "rs_nifty_5d",
    "volume_surge_ratio",
    "dist_sma20",
    "dist_sma50",
    "rsi_14",
    "atr_ratio",
    "bb_bandwidth",
    "bb_percent",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
]

# ==============================================================================
# 1. BENCHMARK INGESTION (NIFTY 50)
# ==============================================================================
def fetch_nifty_benchmark(period: str = "2y") -> pd.DataFrame:
    try:
        nifty = yf.download("^NSEI", period=period, interval="1d", progress=False, auto_adjust=True)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        nifty.index = pd.to_datetime(nifty.index.date)
        nifty["nifty_ret_1d"] = nifty["Close"].pct_change()
        nifty["nifty_ret_5d"] = nifty["Close"].pct_change(5)
        nifty["nifty_sma20"] = nifty["Close"].rolling(20).mean()
        nifty["nifty_bullish"] = (nifty["Close"] > nifty["nifty_sma20"]).astype(float)
        return nifty[["nifty_ret_1d", "nifty_ret_5d", "nifty_bullish"]]
    except Exception as e:
        print(f"Warning: Failed to fetch Nifty benchmark: {e}")
        return pd.DataFrame()


# ==============================================================================
# 2. FEATURE ENGINEERING & REALISTIC TRADEABLE TARGET
# ==============================================================================
def compute_features_and_target(df: pd.DataFrame, nifty_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index.date)

    # Clean alignment with NIFTY 50 by date
    if not nifty_df.empty:
        df = df.join(nifty_df, how="left")
        df["nifty_ret_1d"] = df["nifty_ret_1d"].ffill().fillna(0.0)
        df["nifty_ret_5d"] = df["nifty_ret_5d"].ffill().fillna(0.0)
    else:
        df["nifty_ret_1d"] = 0.0
        df["nifty_ret_5d"] = 0.0

    # Returns & Relative Strength vs Market
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["rs_nifty_1d"] = df["return_1d"] - df["nifty_ret_1d"]
    df["rs_nifty_5d"] = df["return_5d"] - df["nifty_ret_5d"]

    # Volume & Turnover
    vol_sma20 = df["Volume"].rolling(20).mean()
    df["volume_surge_ratio"] = df["Volume"] / (vol_sma20 + 1e-9)
    df["turnover_20d_median"] = (df["Close"] * df["Volume"]).rolling(20).median()

    # Trend Indicators
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["dist_sma20"] = (df["Close"] - df["sma20"]) / (df["sma20"] + 1e-9)
    df["dist_sma50"] = (df["Close"] - df["sma50"]) / (df["sma50"] + 1e-9)

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # ATR (14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr_14"] / (df["Close"] + 1e-9)

    # Bollinger Bands
    rolling_std = df["Close"].rolling(20).std()
    bb_upper = df["sma20"] + (rolling_std * 2.0)
    bb_lower = df["sma20"] - (rolling_std * 2.0)
    df["bb_bandwidth"] = (bb_upper - bb_lower) / (df["sma20"] + 1e-9)
    df["bb_percent"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # Candlestick Anatomy
    candle_range = (df["High"] - df["Low"]) + 1e-9
    df["body_ratio"] = (df["Close"] - df["Open"]).abs() / candle_range
    df["upper_shadow_ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / candle_range
    df["lower_shadow_ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / candle_range

    # REALISTIC TRADEABLE TARGET: Open[t+1] to High[t+1] >= 5% OR Open[t+1] to Close[t+1] >= 4%
    next_open = df["Open"].shift(-1)
    next_high = df["High"].shift(-1)
    next_close = df["Close"].shift(-1)
    open_to_high = (next_high - next_open) / (next_open + 1e-9)
    open_to_close = (next_close - next_open) / (next_open + 1e-9)

    df["target"] = ((open_to_high >= 0.05) | (open_to_close >= 0.04)).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ==============================================================================
# 3. CONCURRENT TICKER WORKER
# ==============================================================================
def fetch_single_ticker(ticker: str, nifty_df: pd.DataFrame, period: str = "2y", max_retries: int = 3):
    yf_symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.04, 0.10))
            df = yf.download(yf_symbol, period=period, interval="1d", progress=False, auto_adjust=True)

            if df.empty or len(df) < 80:
                return None, None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Minimum price >= 10 INR
            latest_price = float(df["Close"].iloc[-1])
            if latest_price < 10.0:
                return None, None

            df = compute_features_and_target(df, nifty_df)

            # Minimum 20-day median turnover >= ₹50 Lakhs (5M INR)
            median_turnover = float(df["turnover_20d_median"].iloc[-1])
            if pd.isna(median_turnover) or median_turnover < 5_000_000:
                return None, None

            latest_row = df.iloc[-1].copy()
            latest_row["Ticker"] = ticker

            train_df = df.iloc[:-1].dropna(subset=FEATURE_COLS + ["target"])
            train_df["Ticker"] = ticker

            return train_df, latest_row
        except Exception:
            time.sleep(0.8 * (attempt + 1))

    return None, None


# ==============================================================================
# 4. SHARD RUNNER
# ==============================================================================
def run_shard_download(ticker_file: str, shard_index: int, total_shards: int, max_workers: int = 6):
    print(f"=== Initializing Institutional Shard {shard_index + 1}/{total_shards} ===")
    nifty_df = fetch_nifty_benchmark(period="2y")

    with open(ticker_file, "r") as f:
        all_tickers = sorted([line.strip().upper() for line in f if line.strip() and not line.startswith("#")])

    my_tickers = all_tickers[shard_index::total_shards]
    print(f"Shard {shard_index} assigned {len(my_tickers)} tickers.")

    shard_train_data = []
    shard_latest_rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_single_ticker, sym, nifty_df): sym for sym in my_tickers}
        completed = 0
        for future in as_completed(future_to_ticker):
            completed += 1
            train_df, latest_row = future.result()
            if train_df is not None and latest_row is not None:
                shard_train_data.append(train_df)
                shard_latest_rows.append(latest_row)
            if completed % 50 == 0 or completed == len(my_tickers):
                print(f"Shard {shard_index}: Filtered & processed [{completed}/{len(my_tickers)}] tickers.")

    os.makedirs("artifacts/shards", exist_ok=True)
    if shard_train_data:
        full_shard_df = pd.concat(shard_train_data, axis=0)
        full_shard_df.to_parquet(f"artifacts/shards/train_shard_{shard_index}.parquet")

    if shard_latest_rows:
        latest_shard_df = pd.DataFrame(shard_latest_rows)
        latest_shard_df.to_parquet(f"artifacts/shards/latest_shard_{shard_index}.parquet")

    print(f"✓ Shard {shard_index} completed. Preserved {len(shard_train_data)} liquid tickers.")


# ==============================================================================
# 5. HIGH-ACCURACY ENSEMBLE TRAINING & SCREENING
# ==============================================================================
def run_train_and_screen():
    print("\n=== Aggregating High-Quality Shard Parquets ===")
    train_files = sorted(glob.glob("artifacts/shards/train_shard_*.parquet"))
    latest_files = sorted(glob.glob("artifacts/shards/latest_shard_*.parquet"))

    if not train_files or not latest_files:
        raise FileNotFoundError("No shard files found in artifacts/shards/ to aggregate.")

    train_data = pd.concat([pd.read_parquet(f) for f in train_files], axis=0).sort_index()
    latest_snapshots = pd.concat([pd.read_parquet(f) for f in latest_files], axis=0).set_index("Ticker")

    train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["target"])

    if len(train_data) > 250000:
        print(f"Retaining latest 250,000 liquid market bars for regime relevance.")
        train_data = train_data.iloc[-250000:]

    X = train_data[FEATURE_COLS].values
    y = train_data["target"].values
    pos_rate = np.mean(y) * 100
    print(f"Institutional Dataset: {X.shape[0]:,} rows across {len(latest_snapshots)} liquid stocks. Tradeable Surge Rate: {pos_rate:.2f}%")

    # TimeSeries K-Partitioning (K=4)
    tscv = TimeSeriesSplit(n_splits=4)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    pipelines = {
        "hist_gb": {
            "pipe": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("clf", HistGradientBoostingClassifier(class_weight="balanced", max_iter=100, random_state=42)),
            ]),
            "params": {
                "clf__learning_rate": list((0.04, 0.08)),
                "clf__max_leaf_nodes": list((15, 31)),
            },
        },
        "random_forest": {
            "pipe": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("clf", RandomForestClassifier(class_weight="balanced", n_estimators=80, max_depth=10, max_samples=0.25, n_jobs=2, random_state=42)),
            ]),
            "params": {
                "clf__min_samples_split": list((20, 50)),
            },
        },
        "extra_trees": {
            "pipe": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("clf", ExtraTreesClassifier(class_weight="balanced", n_estimators=80, max_depth=10, max_samples=0.25, bootstrap=True, n_jobs=2, random_state=42)),
            ]),
            "params": {
                "clf__min_samples_split": list((20, 50)),
            },
        },
        "logistic_reg": {
            "pipe": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)),
            ]),
            "params": {
                "clf__C": list((0.1, 1.0)),
            },
        },
    }

    inner_cv = TimeSeriesSplit(n_splits=2)
    eval_weights = []
    estimators = []

    print("\n--- Tuning Models with Robust Imputation & Cross-Validation ---")
    for name, config in pipelines.items():
        t0 = time.time()
        search = RandomizedSearchCV(
            estimator=config["pipe"],
            param_distributions=config["params"],
            n_iter=2,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=2,
            random_state=42,
        )
        search.fit(X_train, y_train)

        raw_probs = search.best_estimator_.predict_proba(X_test)
        probs = np.take(raw_probs, 1, axis=1)
        preds = (probs >= 0.50).astype(int)

        roc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        elapsed = time.time() - t0
        print(f"[{name:<14}] Fit in {elapsed:.1f}s | ROC-AUC: {roc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        estimators.append((name, search.best_estimator_))
        eval_weights.append(max(roc - 0.50, 0.05))

    total_w = sum(eval_weights)
    norm_weights = [w / total_w for w in eval_weights]
    ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=norm_weights)
    ensemble.fit(X_train, y_train)

    raw_ens_probs = ensemble.predict_proba(X_test)
    ens_probs = np.take(raw_ens_probs, 1, axis=1)
    ens_roc = roc_auc_score(y_test, ens_probs) if len(np.unique(y_test)) > 1 else 0.5
    print(f"\nFinal Ensemble Holdout ROC-AUC: {ens_roc:.4f}")

    # Forward-Looking Inference
    print("\n--- Running Inference for Tomorrow's Market Gainers ---")
    imputer = SimpleImputer(strategy="median")
    imputed_features = imputer.fit_transform(latest_snapshots[FEATURE_COLS].values)
    raw_predictions = ensemble.predict_proba(imputed_features)
    predictions = np.take(raw_predictions, 1, axis=1)

    latest_snapshots["Surge_Probability"] = predictions
    latest_snapshots["Target_5pct"] = latest_snapshots["Close"] * 1.05
    latest_snapshots["Stop_Loss"] = latest_snapshots["Close"] - (latest_snapshots["atr_14"] * 1.5)

    ranked = latest_snapshots.sort_values(by="Surge_Probability", ascending=False).reset_index()
    output_cols = [
        "Ticker", "Close", "Surge_Probability", "Target_5pct", "Stop_Loss",
        "volume_surge_ratio", "rs_nifty_1d", "rsi_14"
    ]
    top_picks = ranked[output_cols].head(25)

    os.makedirs("artifacts", exist_ok=True)
    top_picks.to_csv("artifacts/top_gainers_tomorrow.csv", index=False)
    print("Saved predictions to artifacts/top_gainers_tomorrow.csv")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## 🚀 Tomorrow's Top Institutional NSE Market Gainers (>5% Tradeable Forecast)\n\n")
            f.write(f"- **Liquid Universe:** {len(latest_snapshots)} stocks (Turnover $\ge$ ₹50L, Price $\ge$ ₹10)\n")
            f.write(f"- **Ensemble Holdout ROC-AUC:** {ens_roc:.4f}\n\n")
            f.write("| Ticker | CMP (₹) | Surge Prob | Target (+5%) | Stop Loss (1.5x ATR) | Vol Surge | RS vs NIFTY | RSI (14) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for _, r in top_picks.iterrows():
                f.write(
                    f"| **{r['Ticker']}** | {r['Close']:.2f} | **{r['Surge_Probability']:.1%}** | "
                    f"₹{r['Target_5pct']:.2f} | ₹{r['Stop_Loss']:.2f} | {r['volume_surge_ratio']:.2f}x | {r['rs_nifty_1d']:+.2%} | {r['rsi_14']:.1f} |\n"
                )


# ==============================================================================
# 6. CLI INTERFACE
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Institutional Sharded NSE ML Screener")
    parser.add_argument("--mode", choices=["download", "train", "all"], default="all")
    parser.add_argument("--shard-index", type=int, default=int(os.getenv("SHARD_INDEX", 0)))
    parser.add_argument("--total-shards", type=int, default=int(os.getenv("TOTAL_SHARDS", 1)))
    parser.add_argument("--tickers", type=str, default="tickers.txt")
    parser.add_argument("--workers", type=int, default=6)

    args = parser.parse_args()

    if args.mode == "download":
        run_shard_download(args.tickers, args.shard_index, args.total_shards, max_workers=args.workers)
    elif args.mode == "train":
        run_train_and_screen()
    else:
        run_shard_download(args.tickers, 0, 1, max_workers=args.workers)
        run_train_and_screen()
