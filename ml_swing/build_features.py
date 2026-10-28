import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PARQUET = os.path.join(os.path.dirname(__file__), "dataset.parquet")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "dataset.csv")

def load_nifty50():
    nifty_path = os.path.join(DATA_DIR, "NIFTY50_daily.csv")
    if not os.path.exists(nifty_path):
        raise FileNotFoundError(f"NIFTY 50 data not found at {nifty_path}. Run download_data.py first.")
    df = pd.read_csv(nifty_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    return df

def process_features_and_target():
    nifty_df = load_nifty50()
    nifty_dates = set(nifty_df["date"])
    n_nifty_days = len(nifty_df)
    print(f"[Pipeline] NIFTY 50 benchmark has {n_nifty_days} trading days.")

    # Compute NIFTY 3M return and 5-day forward return
    nifty_df["nifty_mom_3m"] = nifty_df["close"] / nifty_df["close"].shift(63) - 1.0
    nifty_df["nifty_fwd_5d"] = nifty_df["close"].shift(-5) / nifty_df["close"].shift(0) - 1.0
    nifty_lookup = nifty_df.set_index("date")[["nifty_mom_3m", "nifty_fwd_5d", "close"]].rename(
        columns={"close": "nifty_close"}
    )

    stock_files = glob.glob(os.path.join(DATA_DIR, "*_daily.csv"))
    stock_files = [f for f in stock_files if not os.path.basename(f).startswith("NIFTY50")]
    print(f"[Pipeline] Found {len(stock_files)} stock data files to process.")

    combined_rows = []
    dropped_stocks = []
    price_jump_logs = []

    for file_path in stock_files:
        symbol = os.path.basename(file_path).replace("_daily.csv", "")
        df = pd.read_csv(file_path)
        if df.empty or "date" not in df.columns:
            dropped_stocks.append((symbol, "Empty file"))
            continue

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Missing days check: Drop any stock with > 30 missing days relative to Nifty 50
        stock_dates = set(df["date"])
        missing_days = len(nifty_dates - stock_dates)
        if missing_days > 30:
            dropped_stocks.append((symbol, f"Missing {missing_days} days (> 30 allowed)"))
            continue

        # Merge with Nifty benchmark on date (strictly matching dates)
        df = pd.merge(df, nifty_lookup, on="date", how="inner").sort_values("date").reset_index(drop=True)

        # Split/bonus check: Log daily price jumps > 50%
        daily_pct_change = (df["close"] / df["close"].shift(1) - 1.0).abs()
        large_jumps = df[daily_pct_change > 0.50]
        for _, row in large_jumps.iterrows():
            idx = row.name
            prev_close = df.loc[idx - 1, "close"] if idx > 0 else np.nan
            pct = daily_pct_change.loc[idx] * 100
            price_jump_logs.append(f"{symbol} on {row['date']}: price jump of {pct:.1f}% ({prev_close} -> {row['close']})")

        # Feature Engineering (strictly using data up to close of that day - NO LOOKAHEAD)
        # 1. Momentum: 1M (21d), 3M (63d), 6M (126d), 12M (252d)
        df["momentum_1m"] = df["close"] / df["close"].shift(21) - 1.0
        df["momentum_3m"] = df["close"] / df["close"].shift(63) - 1.0
        df["momentum_6m"] = df["close"] / df["close"].shift(126) - 1.0
        df["momentum_12m"] = df["close"] / df["close"].shift(252) - 1.0

        # 2. 20-day historical volatility (annualized)
        daily_ret = df["close"].pct_change()
        df["volatility_20d"] = daily_ret.rolling(window=20).std() * np.sqrt(252)

        # 3. Volume % change (current volume vs 20-day avg)
        vol_ma20 = df["volume"].rolling(window=20).mean()
        # Avoid division by zero
        df["volume_pct_change"] = np.where(vol_ma20 > 0, (df["volume"] - vol_ma20) / vol_ma20, 0.0)

        # 4. Relative strength vs Nifty 50 (stock 3M return minus Nifty 3M return)
        df["rs_vs_nifty_3m"] = df["momentum_3m"] - df["nifty_mom_3m"]

        # 5. Distance from 52-week high (close / rolling_max_252d)
        rolling_max_252d = df["high"].rolling(window=252).max()
        df["dist_from_52w_high"] = df["close"] / rolling_max_252d

        # Target Variable: Look forward 5 trading days
        # Target = 1 if stock's 5-day forward return > Nifty's 5-day forward return, else 0
        df["stock_fwd_5d"] = df["close"].shift(-5) / df["close"] - 1.0
        # Determine target
        df["target"] = (df["stock_fwd_5d"] > df["nifty_fwd_5d"]).astype(int)

        # Data Cleaning:
        # Drop first 252 days (since we need 52-week lookback for 12M momentum and 52w high)
        df_cleaned = df.iloc[252:].copy()

        # Drop the last 5 days where forward return / target cannot be calculated
        df_cleaned = df_cleaned.iloc[:-5].copy()

        df_cleaned["symbol"] = symbol

        feature_cols = [
            "date",
            "symbol",
            "momentum_1m",
            "momentum_3m",
            "momentum_6m",
            "momentum_12m",
            "volatility_20d",
            "volume_pct_change",
            "rs_vs_nifty_3m",
            "dist_from_52w_high",
            "target"
        ]

        df_subset = df_cleaned[feature_cols].dropna().reset_index(drop=True)
        if not df_subset.empty:
            combined_rows.append(df_subset)

    print(f"\n[Cleaning] Stocks dropped due to missing days / insufficient history ({len(dropped_stocks)}):")
    for sym, reason in dropped_stocks:
        print(f"  - {sym}: {reason}")

    if price_jump_logs:
        print(f"\n[Warning] Price jumps > 50% in one day ({len(price_jump_logs)} occurrences):")
        for log in price_jump_logs[:10]:  # print up to 10
            print(f"  - {log}")
        if len(price_jump_logs) > 10:
            print(f"  ... and {len(price_jump_logs) - 10} more.")
    else:
        print("\n[Audit] No price jumps > 50% found across all stocks.")

    if not combined_rows:
        print("\nERROR: No valid data rows generated!")
        return None

    final_df = pd.concat(combined_rows, ignore_index=True)
    # Sort chronologically, then by symbol
    final_df = final_df.sort_values(by=["date", "symbol"]).reset_index(drop=True)

    # Save to Parquet (or CSV if Parquet fails)
    saved = False
    try:
        final_df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"\n[Output] Successfully saved Parquet dataset to: {OUTPUT_PARQUET}")
        saved = True
    except Exception as e:
        print(f"\n[Output] Parquet export failed ({e}). Falling back to CSV...")
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"[Output] Successfully saved CSV dataset to: {OUTPUT_CSV}")
        saved = True

    # Also save CSV as convenient backup
    if saved and not os.path.exists(OUTPUT_CSV):
        final_df.to_csv(OUTPUT_CSV, index=False)

    # Metrics required by prompt
    shape = final_df.shape
    pos_targets = final_df["target"].sum()
    pos_pct = (final_df["target"].mean()) * 100

    print("=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Final Dataset Shape: {shape[0]} rows x {shape[1]} columns")
    print(f"Number of Unique Stocks: {final_df['symbol'].nunique()}")
    print(f"Date Range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"Positive Targets (1): {pos_targets} ({pos_pct:.2f}%)")
    print(f"Negative Targets (0): {len(final_df) - pos_targets} ({100 - pos_pct:.2f}%)")
    print(f"Columns: {list(final_df.columns)}")
    print("=" * 60)
    print("\nSample rows (first 3):")
    print(final_df.head(3).to_string())
    print("\nSample rows (last 3):")
    print(final_df.tail(3).to_string())
    print("=" * 60)

    return final_df

if __name__ == "__main__":
    process_features_and_target()
