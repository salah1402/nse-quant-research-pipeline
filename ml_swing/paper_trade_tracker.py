#!/usr/bin/env python3
"""
Forward Paper Trading System for NSE Multi-Factor Momentum + Delivery Model.

Tasks:
1. Calculate the 50-day SMA of Nifty 50 using nselib on demand.
2. If Nifty < 50 SMA:
   Print "STATUS: BEAR MARKET. GO TO CASH." and exit.
   (Pass --force to bypass the regime filter and generate the Top 10 portfolio anyway).
3. If Nifty > 50 SMA (or --force):
   Fetch the latest Bhavcopy with delivery data on demand using nselib.
   Calculate 12M Momentum Z-score and 20-day Delivery Z-score for the 77-stock universe.
   Rank stocks by Combined Score = Z(Mom) + Z(Delivery).
   Select the Top 10 stocks and save to ml_swing/live_portfolio.csv with equal weights (10% each).
4. Paper Trade Tracking Log:
   Append the rebalance date and the 10 selected stocks to ml_swing/paper_trade_log.csv.
   Provides fields to track Day t+1 Open (entry) and Day t+21 Open (exit) returns.
"""

import os
import sys
import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import nselib
from nselib import capital_market

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data", "bhavcopy_live")
LIVE_PORTFOLIO_CSV = os.path.join(BASE_DIR, "live_portfolio.csv")
PAPER_TRADE_LOG_CSV = os.path.join(BASE_DIR, "paper_trade_log.csv")

# Canonical 77-stock universe
UNIVERSE = [
    'ABB', 'ADANIENT', 'ADANIPORTS', 'ADANIPOWER', 'AMBUJACEM',
    'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJAJFINSV',
    'BAJAJHLDNG', 'BAJFINANCE', 'BANKBARODA', 'BEL', 'BHARTIARTL',
    'BOSCHLTD', 'BPCL', 'BRITANNIA', 'CANBK', 'CGPOWER',
    'CHOLAFIN', 'CIPLA', 'COALINDIA', 'CUMMINSIND', 'DIVISLAB',
    'DLF', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GODREJCP',
    'GRASIM', 'HCLTECH', 'HDFCBANK', 'HINDALCO', 'HINDUNILVR',
    'HINDZINC', 'ICICIBANK', 'IDEA', 'INFY', 'IOC',
    'ITC', 'JINDALSTEL', 'JSWSTEEL', 'KOTAKBANK', 'LT',
    'M&M', 'MARUTI', 'MOTHERSON', 'MUTHOOTFIN', 'NESTLEIND',
    'NTPC', 'ONGC', 'PFC', 'PIDILITIND', 'PNB',
    'POWERGRID', 'RELIANCE', 'SBIN', 'SHRIRAMFIN', 'SIEMENS',
    'SOLARINDS', 'SUNPHARMA', 'TATACONSUM', 'TATAPOWER', 'TATASTEEL',
    'TCS', 'TECHM', 'TITAN', 'TMPV', 'TORNTPHARM',
    'TRENT', 'TVSMOTOR', 'ULTRACEMCO', 'UNIONBANK', 'VEDL',
    'WIPRO', 'ZYDUSLIFE'
]
PORTFOLIO_SIZE = 10
EQUAL_WEIGHT = 1.0 / PORTFOLIO_SIZE  # 10% each


def fetch_nifty_regime(target_date: datetime.date = None):
    """
    Fetch historical Nifty 50 data and calculate the 50-day SMA.
    Returns: (nifty_close, nifty_sma50, is_bull, last_trade_date_str)
    """
    if target_date is None:
        target_date = datetime.date.today()

    # Query ~90 calendar days back (~60 trading days) to stay within NSE's 70-row pagination limit
    # while ensuring at least 50 trading days for the 50-day SMA
    from_date_dt = target_date - datetime.timedelta(days=90)
    from_date_str = from_date_dt.strftime("%d-%m-%Y")
    to_date_str = target_date.strftime("%d-%m-%Y")

    print(f"[1/4] Fetching Nifty 50 index data ({from_date_str} to {to_date_str}) via nselib...", flush=True)
    try:
        df_nifty = capital_market.capital_market_data.get_index_data("NIFTY 50", from_date_str, to_date_str)
    except Exception as e:
        print(f"[Warning] get_index_data direct call failed: {e}. Falling back to period='6M'...", flush=True)
        df_nifty = capital_market.index_data("NIFTY 50", period="6M")

    if df_nifty.empty:
        raise RuntimeError("Failed to retrieve Nifty 50 index data from NSE.")

    df_nifty["trade_date"] = pd.to_datetime(df_nifty["TIMESTAMP"], format="%d-%b-%Y")
    df_nifty["close"] = df_nifty["CLOSE_INDEX_VAL"].astype(float)
    df_nifty = df_nifty.sort_values("trade_date").drop_duplicates(subset=["trade_date"]).reset_index(drop=True)

    df_nifty["sma_50"] = df_nifty["close"].rolling(50, min_periods=50).mean()

    latest_row = df_nifty.dropna(subset=["sma_50"]).iloc[-1]
    nifty_close = float(latest_row["close"])
    nifty_sma50 = float(latest_row["sma_50"])
    trade_date_str = latest_row["trade_date"].strftime("%d-%m-%Y")
    is_bull = nifty_close > nifty_sma50

    return nifty_close, nifty_sma50, is_bull, trade_date_str, latest_row["trade_date"]


def fetch_bhavcopy_for_date(date_str: str) -> pd.DataFrame:
    """
    Fetch Bhavcopy with delivery for a given date (DD-MM-YYYY), filtering to UNIVERSE.
    Uses local parquet caching to avoid redundant NSE network requests.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"bhav_{date_str}.parquet")

    if os.path.exists(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            if not cached_df.empty:
                return cached_df
        except Exception:
            pass

    try:
        raw_df = capital_market.bhav_copy_with_delivery(date_str)
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        # Filter to Equity series and 77 Universe
        df_eq = raw_df[raw_df["SERIES"].isin(["EQ", "BE"])].copy()
        df_filtered = df_eq[df_eq["SYMBOL"].isin(UNIVERSE)].copy()

        # Clean numeric columns
        for col in ["CLOSE_PRICE", "OPEN_PRICE", "TTL_TRD_QNTY", "DELIV_QTY", "DELIV_PER"]:
            if col in df_filtered.columns:
                df_filtered[col] = pd.to_numeric(
                    df_filtered[col].astype(str).str.replace(",", "").str.strip(),
                    errors="coerce"
                )

        df_filtered["trade_date"] = pd.to_datetime(date_str, format="%d-%m-%Y")
        df_filtered.to_parquet(cache_path, index=False)
        return df_filtered
    except Exception as e:
        print(f"[Warning] Failed to fetch Bhavcopy for {date_str}: {e}", flush=True)
        return pd.DataFrame()


def get_recent_trading_dates(n_dates: int = 21) -> list:
    """Get the latest n trading dates from Nifty index history in DD-MM-YYYY format."""
    try:
        df_idx = capital_market.index_data("NIFTY 50", period="1M")
        raw_dates = df_idx["TIMESTAMP"].tolist()
        # Parse and sort descending
        parsed = sorted([datetime.datetime.strptime(d, "%d-%b-%Y") for d in raw_dates], reverse=True)
        return [d.strftime("%d-%m-%Y") for d in parsed[:n_dates]]
    except Exception:
        # Fallback to weekday generation
        dates = []
        curr = datetime.date.today()
        while len(dates) < n_dates:
            if curr.weekday() < 5:
                dates.append(curr.strftime("%d-%m-%Y"))
            curr -= datetime.timedelta(days=1)
        return dates


def fetch_delivery_and_momentum_data(latest_date_str: str):
    """
    Fetch the last 20 trading days of delivery data to calculate Delivery Z-score,
    and fetch 1-year ago Bhavcopy to calculate 12M Momentum.
    """
    print("[2/4] Identifying recent trading dates for rolling delivery window...", flush=True)
    recent_dates = get_recent_trading_dates(21)
    if latest_date_str not in recent_dates:
        recent_dates = [latest_date_str] + [d for d in recent_dates if d != latest_date_str][:20]
    else:
        # Put latest_date_str first
        recent_dates = [latest_date_str] + [d for d in recent_dates if d != latest_date_str]
    recent_dates = recent_dates[:21]

    print(f"      Collecting {len(recent_dates)} trading days of Bhavcopy delivery data...", flush=True)

    dfs = []
    # Concurrently fetch Bhavcopy records with small thread pool
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_bhavcopy_for_date, d): d for d in recent_dates}
        for future in as_completed(futures):
            res = future.result()
            if not res.empty:
                dfs.append(res)

    if not dfs:
        raise RuntimeError("No Bhavcopy data could be downloaded.")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(["SYMBOL", "trade_date"]).reset_index(drop=True)

    # Calculate Delivery Z-Score for each stock
    print("[3/4] Calculating 20-day rolling Delivery Z-scores...", flush=True)
    # delivery_pct is in DELIV_PER column
    stats = combined_df.groupby("SYMBOL")["DELIV_PER"].agg(["mean", "std", "count"]).reset_index()
    stats.columns = ["SYMBOL", "deliv_mean_20d", "deliv_std_20d", "deliv_count"]

    # Latest day snapshot
    latest_dt = datetime.datetime.strptime(latest_date_str, "%d-%m-%Y")
    latest_day_df = combined_df[combined_df["trade_date"] == latest_dt].copy()

    merged_stats = pd.merge(latest_day_df, stats, on="SYMBOL", how="left")
    merged_stats["delivery_zscore"] = np.where(
        merged_stats["deliv_std_20d"] > 0,
        (merged_stats["DELIV_PER"] - merged_stats["deliv_mean_20d"]) / merged_stats["deliv_std_20d"],
        0.0
    )

    # 12-Month Momentum calculation: Close[today] / Close[1_year_ago] - 1.0
    print("      Fetching 1-year ago Bhavcopy for 12M Momentum baseline...", flush=True)
    one_year_ago_dt = latest_dt - datetime.timedelta(days=365)
    # Find nearest trading day around 1 year ago
    hist_bhav = pd.DataFrame()
    for offset in range(5):
        cand_date = (one_year_ago_dt - datetime.timedelta(days=offset)).strftime("%d-%m-%Y")
        hist_bhav = fetch_bhavcopy_for_date(cand_date)
        if not hist_bhav.empty and len(hist_bhav) >= 50:
            break

    if hist_bhav.empty:
        print("[Warning] Could not fetch 1-year ago Bhavcopy. Using zero momentum baseline.", flush=True)
        merged_stats["momentum_12m"] = 0.0
    else:
        hist_close = hist_bhav[["SYMBOL", "CLOSE_PRICE"]].rename(columns={"CLOSE_PRICE": "CLOSE_1Y_AGO"})
        merged_stats = pd.merge(merged_stats, hist_close, on="SYMBOL", how="left")
        merged_stats["momentum_12m"] = np.where(
            merged_stats["CLOSE_1Y_AGO"] > 0,
            (merged_stats["CLOSE_PRICE"] / merged_stats["CLOSE_1Y_AGO"]) - 1.0,
            0.0
        )

    # Fill any remaining NaNs
    merged_stats["momentum_12m"] = merged_stats["momentum_12m"].fillna(0.0)
    merged_stats["delivery_zscore"] = merged_stats["delivery_zscore"].fillna(0.0)

    return merged_stats


def select_top_10_portfolio(stock_df: pd.DataFrame, rebal_date_str: str) -> pd.DataFrame:
    """
    Rank stocks by Combined Score = Z(Mom) + Z(Delivery), select Top 10, equal weight.
    """
    df = stock_df.copy()
    z_mom = (df["momentum_12m"] - df["momentum_12m"].mean()) / (df["momentum_12m"].std() + 1e-12)
    z_del = (df["delivery_zscore"] - df["delivery_zscore"].mean()) / (df["delivery_zscore"].std() + 1e-12)
    df["combined_score"] = z_mom + z_del

    top10 = df.sort_values("combined_score", ascending=False).head(PORTFOLIO_SIZE).reset_index(drop=True)
    top10["rank"] = np.arange(1, len(top10) + 1)
    top10["weight"] = EQUAL_WEIGHT

    portfolio_df = pd.DataFrame({
        "rank": top10["rank"],
        "symbol": top10["SYMBOL"],
        "weight": top10["weight"],
        "close_price": top10["CLOSE_PRICE"].round(2),
        "momentum_12m": (top10["momentum_12m"] * 100.0).round(2),
        "delivery_zscore": top10["delivery_zscore"].round(2),
        "combined_score": top10["combined_score"].round(2),
        "rebalance_date": rebal_date_str
    })
    return portfolio_df


def update_paper_trade_log(portfolio_df: pd.DataFrame, rebal_date_str: str):
    """
    Append new portfolio selections to ml_swing/paper_trade_log.csv.
    """
    columns = [
        "rebalance_date", "rank", "symbol", "weight", "entry_open_price",
        "close_price", "momentum_12m", "delivery_zscore", "combined_score",
        "exit_date", "exit_open_price", "return_pct"
    ]

    log_entries = []
    rebal_dt = datetime.datetime.strptime(rebal_date_str, "%Y-%m-%d")
    target_exit_dt = rebal_dt + datetime.timedelta(days=30)  # Approx 21 trading days

    for _, row in portfolio_df.iterrows():
        log_entries.append({
            "rebalance_date": rebal_date_str,
            "rank": int(row["rank"]),
            "symbol": row["symbol"],
            "weight": float(row["weight"]),
            "entry_open_price": "",  # To be checked and filled at Day t+1 Open
            "close_price": float(row["close_price"]),
            "momentum_12m": float(row["momentum_12m"]),
            "delivery_zscore": float(row["delivery_zscore"]),
            "combined_score": float(row["combined_score"]),
            "exit_date": target_exit_dt.strftime("%Y-%m-%d"),
            "exit_open_price": "",  # To be filled at Day t+21 Open
            "return_pct": ""        # Net return after 21 days
        })

    new_log_df = pd.DataFrame(log_entries, columns=columns)

    if os.path.exists(PAPER_TRADE_LOG_CSV):
        try:
            existing_log = pd.read_csv(PAPER_TRADE_LOG_CSV)
            # Avoid duplicate appends for the exact same rebalance date and symbols
            existing_keys = set(zip(existing_log["rebalance_date"].astype(str), existing_log["symbol"].astype(str)))
            filtered_entries = [e for e in log_entries if (e["rebalance_date"], e["symbol"]) not in existing_keys]
            if filtered_entries:
                append_df = pd.DataFrame(filtered_entries, columns=columns)
                append_df.to_csv(PAPER_TRADE_LOG_CSV, mode="a", header=False, index=False)
                print(f"[Log] Appended {len(filtered_entries)} stocks to {PAPER_TRADE_LOG_CSV}")
            else:
                print(f"[Log] Entries for {rebal_date_str} already exist in {PAPER_TRADE_LOG_CSV}. Skipping duplicate append.")
        except Exception as e:
            new_log_df.to_csv(PAPER_TRADE_LOG_CSV, index=False)
            print(f"[Log] Initialized {PAPER_TRADE_LOG_CSV} (error reading previous: {e})")
    else:
        new_log_df.to_csv(PAPER_TRADE_LOG_CSV, index=False)
        print(f"[Log] Initialized paper trading log at {PAPER_TRADE_LOG_CSV}")


def run_paper_trading(force: bool = False):
    """
    Main paper trading execution pipeline.
    """
    print("=" * 80)
    print("      NSE QUANTITATIVE MULTI-FACTOR MODEL: FORWARD PAPER TRADING SYSTEM       ")
    print("=" * 80)

    # Task 1: Regime Filter
    nifty_close, nifty_sma50, is_bull, trade_date_str, trade_date_dt = fetch_nifty_regime()
    spread_pct = ((nifty_close / nifty_sma50) - 1.0) * 100.0

    print("-" * 80)
    print(f"Trading Date     : {trade_date_str} ({trade_date_dt.strftime('%A')})")
    print(f"Nifty 50 Close   : {nifty_close:,.2f}")
    print(f"Nifty 50-day SMA : {nifty_sma50:,.2f}  (Spread: {spread_pct:+.2f}%)")
    print(f"Regime Check     : {'BULL MARKET (Nifty > 50 SMA)' if is_bull else 'BEAR MARKET (Nifty < 50 SMA)'}")
    print("-" * 80 + "\n")

    # If Nifty < 50 SMA, handle bear regime
    if not is_bull:
        print("STATUS: BEAR MARKET. GO TO CASH.")
        if not force:
            print("\n[INFO] Regime filter is active. Pass --force to bypass the regime filter and generate the Top 10 portfolio anyway.")
            sys.exit(0)
        else:
            print("[OVERRIDE] --force flag detected: Bypassing cash regime filter to generate Top 10 tracking portfolio.\n")

    # If Nifty > 50 SMA (or --force): Generate portfolio
    stock_df = fetch_delivery_and_momentum_data(trade_date_str)
    rebal_iso_date = trade_date_dt.strftime("%Y-%m-%d")

    portfolio_df = select_top_10_portfolio(stock_df, rebal_iso_date)

    # Save to live_portfolio.csv
    portfolio_df.to_csv(LIVE_PORTFOLIO_CSV, index=False)
    print(f"\n[Output] Saved Top 10 Portfolio to: {LIVE_PORTFOLIO_CSV}")

    # Task 2: Append to paper_trade_log.csv
    update_paper_trade_log(portfolio_df, rebal_iso_date)

    # Print summary table
    print("\n" + "=" * 80)
    print("                 CURRENT MONTH PAPER TRADING PORTFOLIO                 ")
    print("=" * 80)
    header_fmt = "{:<4} | {:<12} | {:>8} | {:>12} | {:>12} | {:>14} | {:>14}"
    row_fmt    = "{:<4} | {:<12} | {:>7.1f}% | {:>12.2f} | {:>11.1f}% | {:>14.2f} | {:>14.2f}"

    print(header_fmt.format("Rank", "Symbol", "Weight", "Close (Rs)", "12M Mom (%)", "Delivery Z-Score", "Combined Score"))
    print("-" * 88)
    for _, row in portfolio_df.iterrows():
        print(row_fmt.format(
            int(row["rank"]),
            row["symbol"],
            float(row["weight"]) * 100.0,
            float(row["close_price"]),
            float(row["momentum_12m"]),
            float(row["delivery_zscore"]),
            float(row["combined_score"])
        ))
    print("-" * 88)
    print(f"Rebalance Date   : {rebal_iso_date}")
    print(f"Next Review Date : {(trade_date_dt + datetime.timedelta(days=30)).strftime('%Y-%m-%d')} (Day t+21 Open)")
    print("=" * 88 + "\n")
    return portfolio_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Paper Trading Tracker for NSE Multi-Factor Model")
    parser.add_argument("--force", action="store_true", help="Force portfolio generation even if Nifty < 50 SMA")
    args = parser.parse_args()

    run_paper_trading(force=args.force)
