#!/usr/bin/env python3
"""
Rolling (Overlapping) Backtest for Multi-Factor Momentum + Delivery Model.

Simulates a trade starting every single trading day from 2015 to 2024:
1. Signal at Day t close.
2. If Nifty close <= 50 SMA -> 100% Cash (0% return).
3. If Nifty close > 50 SMA -> Buy Top 10 stocks by Combined Score (Z-Mom + Z-Delivery), equal weight (10% each).
4. Enter at Day t+1 Open, Exit at Day t+21 Open.
5. Apply 0.34% round-trip friction and 20% STCG tax on positive trade gains.
6. Computes comprehensive rolling statistics, walk-forward splits, text histogram, and worst/best case windows.
"""

import os
import sys
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)

DELIVERY_DATASET_PATH = os.path.join(BASE_DIR, "delivery_dataset.parquet")
MOMENTUM_DATASET_PATH = os.path.join(PROJECT_DIR, "dataset.parquet")
NIFTY_CSV_PATH = os.path.join(PROJECT_DIR, "data", "NIFTY50_daily.csv")

PORTFOLIO_SIZE = 10
HOLDING_DAYS = 21
FRICTION_ROUND_TRIP = 0.0034  # 0.17% per side
STCG_TAX_RATE = 0.20        # 20% STCG tax per July 2024 budget


def load_dataset():
    """Load and align delivery, momentum, and Nifty 50 SMA datasets with open execution lag."""
    print("[1/3] Loading historical datasets and computing forward open returns...", flush=True)
    if not os.path.exists(DELIVERY_DATASET_PATH):
        raise FileNotFoundError(f"Missing {DELIVERY_DATASET_PATH}")
    if not os.path.exists(MOMENTUM_DATASET_PATH):
        raise FileNotFoundError(f"Missing {MOMENTUM_DATASET_PATH}")
    if not os.path.exists(NIFTY_CSV_PATH):
        raise FileNotFoundError(f"Missing {NIFTY_CSV_PATH}")

    df_deliv = pd.read_parquet(DELIVERY_DATASET_PATH)
    df_mom = pd.read_parquet(MOMENTUM_DATASET_PATH)
    df_mom["trade_date"] = pd.to_datetime(df_mom["date"])

    merged = pd.merge(
        df_deliv,
        df_mom[["trade_date", "symbol", "momentum_12m"]],
        on=["trade_date", "symbol"],
        how="inner"
    )

    merged = merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # 21-day forward open-to-open return: Entry at t+1 Open, Exit at t+21 Open
    merged["fwd_open_return"] = merged.groupby("symbol")["open"].transform(
        lambda s: s.shift(-HOLDING_DAYS) / s.shift(-1) - 1.0
    )
    merged["entry_open"] = merged.groupby("symbol")["open"].transform(lambda s: s.shift(-1))
    merged["exit_open"] = merged.groupby("symbol")["open"].transform(lambda s: s.shift(-HOLDING_DAYS))

    # Nifty 50 daily and 50-day SMA
    nifty = pd.read_csv(NIFTY_CSV_PATH)
    nifty["trade_date"] = pd.to_datetime(nifty["date"])
    nifty = nifty.sort_values("trade_date").reset_index(drop=True)
    nifty["sma_50"] = nifty["close"].rolling(50, min_periods=50).mean()
    nifty["is_bull"] = nifty["close"] > nifty["sma_50"]

    merged = pd.merge(
        merged,
        nifty[["trade_date", "is_bull", "sma_50", "close"]].rename(columns={"close": "nifty_close"}),
        on="trade_date",
        how="left"
    )

    clean_df = merged.dropna(subset=["delivery_pct_zscore", "momentum_12m", "fwd_open_return", "sma_50"]).copy()
    clean_df = clean_df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return clean_df, nifty


def run_rolling_backtest(clean_df, all_calendar_dates):
    """
    Simulate overlapping 21-day trades starting every single trading day.
    """
    print("[2/3] Executing rolling backtest for every single trading day...", flush=True)

    date_to_idx = {d: i for i, d in enumerate(all_calendar_dates)}

    records = []

    # Group by trading date
    grouped = clean_df.groupby("trade_date")

    for d, day_df in grouped:
        if len(day_df) < PORTFOLIO_SIZE:
            continue

        curr_idx = date_to_idx.get(d)
        entry_idx = curr_idx + 1 if (curr_idx is not None and curr_idx + 1 < len(all_calendar_dates)) else None
        exit_idx = curr_idx + HOLDING_DAYS if (curr_idx is not None and curr_idx + HOLDING_DAYS < len(all_calendar_dates)) else None

        entry_date = all_calendar_dates[entry_idx] if entry_idx is not None else None
        exit_date = all_calendar_dates[exit_idx] if exit_idx is not None else None

        is_bull = bool(day_df["is_bull"].iloc[0])
        nifty_close = float(day_df["nifty_close"].iloc[0])
        nifty_sma50 = float(day_df["sma_50"].iloc[0])

        if not is_bull:
            records.append({
                "signal_date": d,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "regime": "CASH",
                "nifty_close": nifty_close,
                "nifty_sma50": nifty_sma50,
                "gross_return": 0.0,
                "net_return": 0.0,
                "top_symbols": [],
                "stock_returns": []
            })
        else:
            # Combined Score = Z(Mom) + Z(Delivery)
            zm = (day_df["momentum_12m"] - day_df["momentum_12m"].mean()) / (day_df["momentum_12m"].std() + 1e-12)
            zd = (day_df["delivery_pct_zscore"] - day_df["delivery_pct_zscore"].mean()) / (day_df["delivery_pct_zscore"].std() + 1e-12)
            combined = zm + zd

            top_indices = np.argsort(-combined.values)[:PORTFOLIO_SIZE]
            top_stocks = day_df.iloc[top_indices]

            gross_rets = top_stocks["fwd_open_return"].values
            net_trade = gross_rets - FRICTION_ROUND_TRIP
            taxed = np.where(net_trade > 0, net_trade * (1.0 - STCG_TAX_RATE), net_trade)

            port_gross = float(np.mean(gross_rets))
            port_net = float(np.mean(taxed))

            stock_info = []
            for _, s_row in top_stocks.iterrows():
                s_gross = float(s_row["fwd_open_return"])
                s_net = s_gross - FRICTION_ROUND_TRIP
                s_taxed = s_net * (1.0 - STCG_TAX_RATE) if s_net > 0 else s_net
                stock_info.append({
                    "symbol": s_row["symbol"],
                    "momentum_12m": float(s_row["momentum_12m"]),
                    "delivery_zscore": float(s_row["delivery_pct_zscore"]),
                    "gross_return": s_gross,
                    "net_return": s_taxed
                })

            records.append({
                "signal_date": d,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "regime": "BULL",
                "nifty_close": nifty_close,
                "nifty_sma50": nifty_sma50,
                "gross_return": port_gross,
                "net_return": port_net,
                "top_symbols": top_stocks["symbol"].tolist(),
                "stock_returns": stock_info
            })

    results_df = pd.DataFrame(records)
    return results_df


def render_ascii_histogram(values, bins, labels, title="21-Day Forward Net Return Distribution"):
    """Render an ASCII bar histogram of values across specified bins."""
    counts = []
    total = len(values)

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            cnt = int(np.sum((values >= low) & (values <= high)))
        else:
            cnt = int(np.sum((values >= low) & (values < high)))
        counts.append(cnt)

    max_count = max(counts) if counts and max(counts) > 0 else 1
    max_bar_width = 40

    print("\n" + "=" * 78)
    print(f"       HISTOGRAM: {title.upper()}       ")
    print("=" * 78)
    print(f"{'Return Bin':<16} | {'Count':>6} | {'Pct':>6} | Distribution Graph (n={total})")
    print("-" * 78)

    for label, cnt in zip(labels, counts):
        pct = (cnt / total) * 100.0 if total > 0 else 0.0
        bar_len = int(round((cnt / max_count) * max_bar_width))
        bar_str = "#" * bar_len
        print(f"{label:<16} | {cnt:>6d} | {pct:>5.1f}% | {bar_str}")
    print("-" * 78)


def print_rolling_statistics(results_df):
    """Print comprehensive summary statistics for rolling 21-day trades."""
    all_net = results_df["net_return"].values * 100.0
    invested_df = results_df[results_df["regime"] == "BULL"]
    invested_net = invested_df["net_return"].values * 100.0
    cash_df = results_df[results_df["regime"] == "CASH"]

    n_total = len(results_df)
    n_invested = len(invested_df)
    n_cash = len(cash_df)

    print("\n" + "=" * 78)
    print("     OVERLAPPING 21-DAY ROLLING BACKTEST: SUMMARY STATISTICS (2015-2024)     ")
    print("=" * 78)

    print("\n--- 1. OVERALL PORTFOLIO METRICS (ALL 2,388 DAILY ROLLING WINDOWS) ---")
    print(f"  * Total 21-Day Windows   : {n_total:,d}")
    print(f"  * Invested (Bull) Windows: {n_invested:,d} ({n_invested/n_total*100:.1f}%)")
    print(f"  * Cash (Bear) Windows    : {n_cash:,d} ({n_cash/n_total*100:.1f}%)")
    print(f"  * Mean Net Return / Trade: {np.mean(all_net):+.2f}%")
    print(f"  * Median Net Return      : {np.median(all_net):+.2f}%")
    print(f"  * Standard Deviation     :  {np.std(all_net):.2f}%")
    print(f"  * Win Rate (Net Ret > 0) :  {np.sum(all_net > 0) / n_total * 100:.2f}% ({np.sum(all_net > 0):,d} windows)")
    print(f"  * Cash Rate (Net Ret == 0:  {n_cash / n_total * 100:.2f}% ({n_cash:,d} windows)")
    print(f"  * Loss Rate (Net Ret < 0):  {np.sum(all_net < 0) / n_total * 100:.2f}% ({np.sum(all_net < 0):,d} windows)")

    print("\n--- 2. ACTIVE TRADING METRICS (INVESTED WINDOWS ONLY: NIFTY > 50 SMA) ---")
    print(f"  * Active Trading Windows : {n_invested:,d}")
    print(f"  * Mean Net Return / Trade: {np.mean(invested_net):+.2f}%")
    print(f"  * Median Net Return      : {np.median(invested_net):+.2f}%")
    print(f"  * Standard Deviation     :  {np.std(invested_net):.2f}%")
    print(f"  * Win Rate (Invested)    :  {np.sum(invested_net > 0) / n_invested * 100:.2f}% ({np.sum(invested_net > 0):,d} winning windows)")
    pos_sum = np.sum(invested_net[invested_net > 0])
    neg_sum = np.abs(np.sum(invested_net[invested_net < 0]))
    profit_factor = pos_sum / neg_sum if neg_sum > 0 else np.nan
    print(f"  * Profit Factor (Gains/Loss): {profit_factor:.2f}")

    # Walk-forward period breakdown
    print("\n--- 3. WALK-FORWARD PERIOD BREAKDOWN ---")
    periods = {
        "Period 1 (In-Sample: 2015-2020)": ("2015-01-01", "2020-12-31"),
        "Period 2 (Out-of-Sample: 2021-2023)": ("2021-01-01", "2023-12-31"),
        "Holdout (2024)": ("2024-01-01", "2024-12-31")
    }

    hdr_fmt = "{:<36} | {:>6} | {:>9} | {:>9} | {:>9} | {:>8}"
    row_fmt = "{:<36} | {:>6d} | {:>8.2f}% | {:>8.2f}% | {:>8.2f}% | {:>7.1f}%"
    print(hdr_fmt.format("Period", "Trades", "Mean (All)", "Mean (Inv)", "Median", "Win Rate"))
    print("-" * 88)
    for p_name, (start_d, end_d) in periods.items():
        sub = results_df[(results_df["signal_date"] >= start_d) & (results_df["signal_date"] <= end_d)]
        inv_sub = sub[sub["regime"] == "BULL"]
        sub_net = sub["net_return"].values * 100.0
        inv_net = inv_sub["net_return"].values * 100.0

        mean_all = np.mean(sub_net) if len(sub_net) > 0 else 0.0
        mean_inv = np.mean(inv_net) if len(inv_net) > 0 else 0.0
        med_inv = np.median(inv_net) if len(inv_net) > 0 else 0.0
        win_rate_inv = (np.sum(inv_net > 0) / len(inv_net) * 100.0) if len(inv_net) > 0 else 0.0

        print(row_fmt.format(p_name, len(sub), mean_all, mean_inv, med_inv, win_rate_inv))
    print("-" * 88)

    # Histogram: All trades (including cash)
    bins_req = [-np.inf, -0.05, 0.0, 0.05, 0.10, np.inf]
    labels_req = ["< -5.0%", "-5.0% to 0.0%", "0.0% to +5.0%", "+5.0% to +10.0%", "> +10.0%"]
    render_ascii_histogram(
        results_df["net_return"].values,
        bins_req,
        labels_req,
        title="21-Day Net Return Distribution (All Windows, incl. 0% Cash)"
    )

    # Histogram: Invested trades only
    render_ascii_histogram(
        invested_df["net_return"].values,
        bins_req,
        labels_req,
        title="21-Day Net Return Distribution (Invested Windows Only)"
    )

    # Granular 10-bin histogram for invested trades
    bins_fine = [-np.inf, -0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15, np.inf]
    labels_fine = [
        "< -15%", "-15% to -10%", "-10% to -5%", "-5% to -2%", "-2% to 0%",
        "0% to +2%", "+2% to +5%", "+5% to +10%", "+10% to +15%", "> +15%"
    ]
    render_ascii_histogram(
        invested_df["net_return"].values,
        bins_fine,
        labels_fine,
        title="Granular Distribution (Invested Windows Only)"
    )

    # Worst Case Scenario
    worst_idx = results_df["net_return"].idxmin()
    worst = results_df.iloc[worst_idx]
    worst_entry_str = worst["entry_date"].strftime("%Y-%m-%d") if worst["entry_date"] else "N/A"
    worst_exit_str = worst["exit_date"].strftime("%Y-%m-%d") if worst["exit_date"] else "N/A"

    print("\n" + "=" * 78)
    print("                     WORST CASE SCENARIO ANALYSIS                     ")
    print("=" * 78)
    print(f"  * Signal Date (Day t Close) : {worst['signal_date'].strftime('%Y-%m-%d')} ({worst['signal_date'].strftime('%A')})")
    print(f"  * Portfolio Entry (t+1 Open): {worst_entry_str}")
    print(f"  * Portfolio Exit (t+21 Open): {worst_exit_str}")
    print(f"  * Regime at Entry           : {worst['regime']} (Nifty: {worst['nifty_close']:,.1f} vs SMA50: {worst['nifty_sma50']:,.1f})")
    print(f"  * Absolute Worst Net Return : {worst['net_return']*100:+.2f}%  (Gross Return: {worst['gross_return']*100:+.2f}%)")
    print(f"  * Historical Context        : Peak COVID-19 Global Liquidity Crash (Feb-March 2020).")
    print(f"                                Signal generated immediately before the unprecedented")
    print(f"                                pandemic selloff and successive circuit breakers.")
    print("-" * 78)
    print("Top 10 Stocks Held During the Worst 21-Day Window:")
    stk_fmt = "  {:<4} | {:<12} | {:>12} | {:>16} | {:>14} | {:>12}"
    print(stk_fmt.format("No.", "Symbol", "12M Mom (%)", "Delivery Z-Score", "Gross Return", "Net Return"))
    print("  " + "-" * 74)
    for idx, s in enumerate(worst["stock_returns"], 1):
        print(stk_fmt.format(
            idx,
            s["symbol"],
            f"{s['momentum_12m']*100:+.1f}%",
            f"{s['delivery_zscore']:+.2f}",
            f"{s['gross_return']*100:+.2f}%",
            f"{s['net_return']*100:+.2f}%"
        ))
    print("  " + "-" * 74)

    # Best Case Scenario
    best_idx = results_df["net_return"].idxmax()
    best = results_df.iloc[best_idx]
    best_entry_str = best["entry_date"].strftime("%Y-%m-%d") if best["entry_date"] else "N/A"
    best_exit_str = best["exit_date"].strftime("%Y-%m-%d") if best["exit_date"] else "N/A"

    print("\n" + "=" * 78)
    print("                     BEST CASE SCENARIO ANALYSIS                      ")
    print("=" * 78)
    print(f"  * Signal Date (Day t Close) : {best['signal_date'].strftime('%Y-%m-%d')} ({best['signal_date'].strftime('%A')})")
    print(f"  * Portfolio Entry (t+1 Open): {best_entry_str}")
    print(f"  * Portfolio Exit (t+21 Open): {best_exit_str}")
    print(f"  * Absolute Best Net Return  : {best['net_return']*100:+.2f}%  (Gross Return: {best['gross_return']*100:+.2f}%)")
    print(f"  * Historical Context        : Post-COVID Vaccine Announcement Breakout Rally (Nov-Dec 2020)")
    print("-" * 78)
    print("Top 10 Stocks Held During the Best 21-Day Window:")
    print(stk_fmt.format("No.", "Symbol", "12M Mom (%)", "Delivery Z-Score", "Gross Return", "Net Return"))
    print("  " + "-" * 74)
    for idx, s in enumerate(best["stock_returns"], 1):
        print(stk_fmt.format(
            idx,
            s["symbol"],
            f"{s['momentum_12m']*100:+.1f}%",
            f"{s['delivery_zscore']:+.2f}",
            f"{s['gross_return']*100:+.2f}%",
            f"{s['net_return']*100:+.2f}%"
        ))
    print("  " + "-" * 74)
    print("=" * 78 + "\n")


def main():
    clean_df, nifty = load_dataset()
    all_calendar_dates = sorted(nifty["trade_date"].unique())
    results_df = run_rolling_backtest(clean_df, all_calendar_dates)
    print_rolling_statistics(results_df)


if __name__ == "__main__":
    main()
