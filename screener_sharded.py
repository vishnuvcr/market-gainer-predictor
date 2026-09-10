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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "return_1d",
    "return_5d",
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
# 1. TECHNICAL INDICATORS & TARGET COMPUTATION
# ==============================================================================
def compute_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)

    # Volume dynamics
    vol_sma20 = df["Volume"].rolling(20).mean()
    df["volume_surge_ratio"] = df["Volume"] / (vol_sma20 + 1e-9)

    # Trend indicators
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

    # Candlestick anatomy
    candle_range = (df["High"] - df["Low"]) + 1e-9
    df["body_ratio"] = (df["Close"] - df["Open"]).abs() / candle_range
    df["upper_shadow_ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / candle_range
    df["lower_shadow_ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / candle_range

    # TARGET: Next day return >= +5%
    next_day_ret = df["Close"].shift(-1) / df["Close"] - 1.0
    df["target"] = (next_day_ret >= 0.05).astype(int)

    return df


# ==============================================================================
# 2. CONCURRENT TICKER WORKER
# ==============================================================================
def fetch_single_ticker(ticker: str, period: str = "2y", max_retries: int = 3):
    yf_symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.05, 0.15))
            df = yf.download(yf_symbol, period=period, interval="1d", progress=False, auto_adjust=True)

            if df.empty or len(df) < 80:
                return None, None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = compute_features_and_target(df)

            latest_row = df.iloc[-1].copy()
            latest_row["Ticker"] = ticker

            train_df = df.iloc[:-1].dropna()
            train_df["Ticker"] = ticker

            return train_df, latest_row
        except Exception:
            time.sleep(1.0 * (attempt + 1))

    return None, None


# ==============================================================================
# 3. SHARD RUNNER
# ==============================================================================
def run_shard_download(ticker_file: str, shard_index: int, total_shards: int, max_workers: int = 6):
    print(f"=== Initializing Shard {shard_index + 1}/{total_shards} ===")

    with open(ticker_file, "r") as f:
        all_tickers = sorted([line.strip().upper() for line in f if line.strip() and not line.startswith("#")])

    my_tickers = all_tickers[shard_index::total_shards]
    print(f"Shard {shard_index} assigned {len(my_tickers)} out of {len(all_tickers)} total tickers.")

    shard_train_data = []
    shard_latest_rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_single_ticker, sym): sym for sym in my_tickers}
        completed = 0
        for future in as_completed(future_to_ticker):
            completed += 1
            train_df, latest_row = future.result()
            if train_df is not None and latest_row is not None:
                shard_train_data.append(train_df)
                shard_latest_rows.append(latest_row)
            if completed % 25 == 0 or completed == len(my_tickers):
                print(f"Shard {shard_index}: Processed [{completed}/{len(my_tickers)}] tickers.")

    os.makedirs("artifacts/shards", exist_ok=True)
    if shard_train_data:
        full_shard_df = pd.concat(shard_train_data, axis=0)
        full_shard_df.to_parquet(f"artifacts/shards/train_shard_{shard_index}.parquet")

    if shard_latest_rows:
        latest_shard_df = pd.DataFrame(shard_latest_rows)
        latest_shard_df.to_parquet(f"artifacts/shards/latest_shard_{shard_index}.parquet")

    print(f"✓ Shard {shard_index} finished. Saved {len(shard_train_data)} active tickers.")


# ==============================================================================
# 4. AGGREGATION, ENSEMBLE TRAINING & SCREENING
# ==============================================================================
def run_train_and_screen():
    print("\n=== Aggregating Shard Parquets ===")
    train_files = sorted(glob.glob("artifacts/shards/train_shard_*.parquet"))
    latest_files = sorted(glob.glob("artifacts/shards/latest_shard_*.parquet"))

    if not train_files or not latest_files:
        raise FileNotFoundError("No shard files found in artifacts/shards/ to aggregate.")

    train_data = pd.concat([pd.read_parquet(f) for f in train_files], axis=0).sort_index()
    latest_snapshots = pd.concat([pd.read_parquet(f) for f in latest_files], axis=0).set_index("Ticker")

    X = train_data[FEATURE_COLS].values
    y = train_data["target"].values
    pos_rate = np.mean(y) * 100
    print(f"Merged Dataset: {X.shape[0]:,} rows across {len(latest_snapshots)} stocks. Surge Target (5%+) Rate: {pos_rate:.2f}%")

    # TimeSeries K-Partitioning (K=5)
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Model Pipelines
    pipelines = {
        "hist_gb": {
            "pipe": Pipeline([
                ("scaler", RobustScaler()),
                ("clf", HistGradientBoostingClassifier(class_weight="balanced", random_state=42)),
            ]),
            "params": {
                "clf__max_iter": [60, 100],
                "clf__learning_rate": [0.03, 0.08],
                "clf__max_leaf_nodes": [15, 31],
            },
        },
        "random_forest": {
            "pipe": Pipeline([
                ("scaler", RobustScaler()),
                ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)),
            ]),
            "params": {
                "clf__n_estimators": [70, 120],
                "clf__max_depth":,
            },
        },
        "extra_trees": {
            "pipe": Pipeline([
                ("scaler", RobustScaler()),
                ("clf", ExtraTreesClassifier(class_weight="balanced", n_jobs=-1, random_state=42)),
            ]),
            "params": {
                "clf__n_estimators": [70, 120],
                "clf__max_depth":,
            },
        },
        "logistic_reg": {
            "pipe": Pipeline([
                ("scaler", RobustScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=400, random_state=42)),
            ]),
            "params": {
                "clf__C": [0.05, 0.5, 2.0],
            },
        },
    }

    # Inner TimeSeriesSplit for Hyperparameter Tuning
    inner_cv = TimeSeriesSplit(n_splits=3)
    eval_weights = []
    estimators = []

    print("\n--- Tuning Models via Time-Partitioned Validation ---")
    for name, config in pipelines.items():
        search = RandomizedSearchCV(
            estimator=config["pipe"],
            param_distributions=config["params"],
            n_iter=4,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            random_state=42,
        )
        search.fit(X_train, y_train)

        # Slice column 1 (positive class probability)
        probs = search.best_estimator_.predict_proba(X_test)
        preds = (probs >= 0.50).astype(int)

        roc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"[{name:<14}] Holdout ROC-AUC: {roc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        estimators.append((name, search.best_estimator_))
        eval_weights.append(max(roc - 0.50, 0.05))

    # Soft-Voting Ensemble
    total_w = sum(eval_weights)
    norm_weights = [w / total_w for w in eval_weights]
    ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=norm_weights)
    ensemble.fit(X_train, y_train)

    # Slice column 1 for ensemble predictions
    ens_probs = ensemble.predict_proba(X_test)
    ens_roc = roc_auc_score(y_test, ens_probs) if len(np.unique(y_test)) > 1 else 0.5
    print(f"\nFinal Ensemble Holdout ROC-AUC: {ens_roc:.4f}")

    # Forward-Looking Inference
    print("\n--- Running Inference for Tomorrow's Market Gainers ---")
    latest_features = latest_snapshots[FEATURE_COLS].astype(float).values
    predictions = ensemble.predict_proba(latest_features)

    latest_snapshots["Surge_Probability"] = predictions
    latest_snapshots["Target_5pct"] = latest_snapshots["Close"] * 1.05
    latest_snapshots["Stop_Loss"] = latest_snapshots["Close"] - (latest_snapshots["atr_14"] * 1.5)

    ranked = latest_snapshots.sort_values(by="Surge_Probability", ascending=False).reset_index()
    output_cols = ["Ticker", "Close", "Surge_Probability", "Target_5pct", "Stop_Loss", "volume_surge_ratio", "rsi_14"]
    top_picks = ranked[output_cols].head(20)

    top_picks.to_csv("artifacts/top_gainers_tomorrow.csv", index=False)
    print("Saved predictions to artifacts/top_gainers_tomorrow.csv")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## 🚀 Tomorrow's Top NSE Market Gainers (>5% Surge Forecast)\n\n")
            f.write(f"- **Universe Scanned:** {len(latest_snapshots)} stocks\n")
            f.write(f"- **Ensemble Holdout ROC-AUC:** {ens_roc:.4f}\n\n")
            f.write("| Ticker | CMP (₹) | Surge Prob | Target (+5%) | Stop Loss (1.5x ATR) | Vol Surge | RSI (14) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for _, r in top_picks.iterrows():
                f.write(
                    f"| **{r['Ticker']}** | {r['Close']:.2f} | **{r['Surge_Probability']:.1%}** | "
                    f"₹{r['Target_5pct']:.2f} | ₹{r['Stop_Loss']:.2f} | {r['volume_surge_ratio']:.2f}x | {r['rsi_14']:.1f} |\n"
                )


# ==============================================================================
# 5. CLI INTERFACE
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Speed Sharded NSE ML Screener")
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
