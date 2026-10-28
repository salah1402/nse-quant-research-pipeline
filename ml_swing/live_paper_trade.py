import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DELIVERY_DATASET_PATH = os.path.join(BASE_DIR, "fundamentals", "delivery_dataset.parquet")
MOMENTUM_DATASET_PATH = os.path.join(BASE_DIR, "dataset.parquet")
NIFTY_CSV_PATH = os.path.join(BASE_DIR, "data", "NIFTY50_daily.csv")
TARGET_PORTFOLIO_CSV = os.path.join(BASE_DIR, "target_portfolio.csv")

PORTFOLIO_SIZE = 10
EQUAL_WEIGHT = 1.0 / PORTFOLIO_SIZE  # 10% each


def load_latest_dataset():
    """Load merged dataset with 12M momentum, delivery z-scores, and Nifty 50 SMA."""
    if not os.path.exists(DELIVERY_DATASET_PATH):
        raise FileNotFoundError(f"Missing {DELIVERY_DATASET_PATH}. Run build_delivery_dataset.py first.")
    df_deliv = pd.read_parquet(DELIVERY_DATASET_PATH)

    if not os.path.exists(MOMENTUM_DATASET_PATH):
        raise FileNotFoundError(f"Missing {MOMENTUM_DATASET_PATH}. Run build_features.py first.")
    df_mom = pd.read_parquet(MOMENTUM_DATASET_PATH)
    df_mom["trade_date"] = pd.to_datetime(df_mom["date"])

    # Merge stock datasets
    merged = pd.merge(
        df_deliv,
        df_mom[["trade_date", "symbol", "momentum_12m"]],
        on=["trade_date", "symbol"],
        how="inner"
    )

    # Nifty 50 50-day SMA
    if not os.path.exists(NIFTY_CSV_PATH):
        raise FileNotFoundError(f"Missing {NIFTY_CSV_PATH}")
    nifty = pd.read_csv(NIFTY_CSV_PATH)
    nifty["trade_date"] = pd.to_datetime(nifty["date"])
    nifty = nifty.sort_values("trade_date").reset_index(drop=True)
    nifty["sma_50"] = nifty["close"].rolling(50).mean()
    nifty["nifty_close"] = nifty["close"]

    merged = pd.merge(
        merged,
        nifty[["trade_date", "nifty_close", "sma_50"]],
        on="trade_date",
        how="left"
    )

    clean_df = merged.dropna(subset=["delivery_pct_zscore", "momentum_12m", "sma_50"]).copy()
    clean_df = clean_df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return clean_df


def generate_portfolio(target_date_str=None):
    clean_df = load_latest_dataset()
    available_dates = sorted(clean_df["trade_date"].unique())

    if target_date_str:
        target_date = pd.Timestamp(target_date_str)
        if target_date not in available_dates:
            # Pick closest previous date
            prior_dates = [d for d in available_dates if d <= target_date]
            if not prior_dates:
                raise ValueError(f"No data available on or before {target_date_str}")
            target_date = prior_dates[-1]
    else:
        target_date = available_dates[-1]

    day_df = clean_df[clean_df["trade_date"] == target_date].copy()
    nifty_close = day_df["nifty_close"].iloc[0]
    nifty_sma50 = day_df["sma_50"].iloc[0]
    is_bull = bool(nifty_close > nifty_sma50)

    print("=" * 80)
    print("           NSE QUANTITATIVE MULTI-FACTOR: MONTHLY PAPER TRADER           ")
    print("=" * 80)
    print(f"Rebalance Date   : {target_date.strftime('%Y-%m-%d')} ({target_date.strftime('%A')})")
    print(f"Universe Size    : {len(day_df)} stocks available")
    print(f"Nifty 50 Close   : {nifty_close:,.2f}")
    print(f"Nifty 50-day SMA : {nifty_sma50:,.2f}  (Spread: {((nifty_close/nifty_sma50 - 1)*100):+.2f}%)")
    print(f"Regime Status    : {'BULLISH (Invested)' if is_bull else 'BEARISH (100% Cash)'}")
    print("=" * 80 + "\n")

    if not is_bull:
        print("[REGIME ALERT] Nifty 50 is BELOW its 50-day SMA.")
        print("[ACTION] Allocating 100% of capital to CASH. No stocks will be purchased.\n")

        cash_df = pd.DataFrame([{
            "rank": 1,
            "symbol": "100% CASH",
            "weight": 1.0,
            "close_price": np.nan,
            "momentum_12m": np.nan,
            "delivery_zscore": np.nan,
            "combined_score": np.nan,
            "status": "BEAR REGIME (Nifty < 50 SMA) - STAY IN CASH"
        }])
        cash_df.to_csv(TARGET_PORTFOLIO_CSV, index=False)
        print(f"[Output] Saved cash allocation to {TARGET_PORTFOLIO_CSV}")
        return cash_df

    # Regime is BULL: Rank by Combined Score = Z(Momentum) + Z(Delivery)
    z_mom = (day_df["momentum_12m"] - day_df["momentum_12m"].mean()) / (day_df["momentum_12m"].std() + 1e-12)
    z_del = (day_df["delivery_pct_zscore"] - day_df["delivery_pct_zscore"].mean()) / (day_df["delivery_pct_zscore"].std() + 1e-12)
    day_df["combined_score"] = z_mom + z_del

    top10 = day_df.sort_values("combined_score", ascending=False).head(PORTFOLIO_SIZE).reset_index(drop=True)
    top10["rank"] = np.arange(1, len(top10) + 1)
    top10["weight"] = EQUAL_WEIGHT

    portfolio_df = top10[[
        "rank", "symbol", "weight", "close", "momentum_12m", "delivery_pct_zscore", "combined_score"
    ]].rename(columns={
        "close": "close_price",
        "delivery_pct_zscore": "delivery_zscore"
    })

    portfolio_df.to_csv(TARGET_PORTFOLIO_CSV, index=False)

    print("--- TARGET PORTFOLIO (Top 10 Stocks, Equal Weight 10.0% Each) ---")
    header_fmt = "{:<4} | {:<12} | {:>8} | {:>12} | {:>12} | {:>15} | {:>14}"
    row_fmt    = "{:<4} | {:<12} | {:>7.1f}% | {:>12.2f} | {:>11.1f}% | {:>15.2f} | {:>14.2f}"

    print(header_fmt.format("Rank", "Symbol", "Weight", "Close (Rs)", "12M Mom (%)", "Delivery Z-Score", "Combined Score"))
    print("-" * 88)
    for _, row in portfolio_df.iterrows():
        print(row_fmt.format(
            int(row["rank"]),
            row["symbol"],
            row["weight"] * 100.0,
            row["close_price"],
            row["momentum_12m"] * 100.0,
            row["delivery_zscore"],
            row["combined_score"]
        ))
    print("-" * 88)
    print(f"\n[Output] Saved target portfolio to: {TARGET_PORTFOLIO_CSV}\n")
    return portfolio_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monthly Paper Trader for Multi-Factor Momentum + Delivery Model")
    parser.add_argument("--date", type=str, default=None, help="Target trade date (YYYY-MM-DD). Defaults to latest available.")
    args = parser.parse_args()
    generate_portfolio(args.date)
