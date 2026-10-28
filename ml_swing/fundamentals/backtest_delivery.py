import os
import time
import numpy as np
import pandas as pd

DATASET_PATH = os.path.join(os.path.dirname(__file__), "delivery_dataset.parquet")

# Costs & Taxes constants
FRICTION_PER_SIDE = 0.0017         # 0.17% per side
ROUND_TRIP_FRICTION = 2 * FRICTION_PER_SIDE  # 0.34% round trip
STCG_TAX_RATE = 0.15               # 15% Short-Term Capital Gains tax on profitable trades
N_STOCKS_SELECT = 5                # Top 5 / Bottom 5
HOLDING_DAYS = 5                   # Non-overlapping 5-day holding period

PERIODS = {
    "Period 1 (In-Sample: 2015-2020)": ("2015-01-01", "2020-12-31"),
    "Period 2 (Out-of-Sample: 2021-2023)": ("2021-01-01", "2023-12-31"),
    "Holdout (2024)": ("2024-01-01", "2024-12-31"),
}


def load_clean_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run build_delivery_dataset.py first.")
    df = pd.read_parquet(DATASET_PATH)
    clean_df = df.dropna(subset=["delivery_pct_zscore", "fwd_5d_return"]).copy()
    clean_df = clean_df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return clean_df


def simulate_period(df_period, rebalance_dates, start_date, end_date):
    """
    Simulate Long-Only and Long-Short 5-day ranking strategies on given rebalance dates.
    """
    years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
    if years <= 0:
        years = 1.0

    # Long-Only tracking
    cap_lo = 1.0
    equity_lo = [cap_lo]
    period_rets_lo = []
    trade_profits_lo = []

    # Long-Short tracking (50% long Top 5, 50% short Bottom 5)
    cap_ls = 1.0
    equity_ls = [cap_ls]
    period_rets_ls = []
    trade_profits_ls = []

    for d in rebalance_dates:
        day_df = df_period[df_period["trade_date"] == d].sort_values("delivery_pct_zscore", ascending=False)
        if len(day_df) < 10:
            continue

        top5 = day_df.head(N_STOCKS_SELECT)
        bot5 = day_df.tail(N_STOCKS_SELECT)

        # ---------------------------------------------------------
        # Long-Only: Buy Top 5, 0.34% round-trip friction, 15% STCG on positive trades
        # ---------------------------------------------------------
        top5_gross = top5["fwd_5d_return"].values
        top5_net_trade = top5_gross - ROUND_TRIP_FRICTION
        # 15% STCG tax applied to positive net profits per trade
        top5_after_tax = np.where(top5_net_trade > 0, top5_net_trade * (1.0 - STCG_TAX_RATE), top5_net_trade)
        
        r_period_lo = np.mean(top5_after_tax)
        cap_lo *= (1.0 + r_period_lo)
        equity_lo.append(cap_lo)
        period_rets_lo.append(r_period_lo)
        trade_profits_lo.extend(top5_after_tax.tolist())

        # ---------------------------------------------------------
        # Long-Short: Buy Top 5, Short Bottom 5
        # 0.34% round-trip friction each side, ignore STCG (futures treatment)
        # Dollar neutral: 50% Long leg + 50% Short leg
        # ---------------------------------------------------------
        long_leg = top5_net_trade  # gross long - 0.0034
        bot5_gross = bot5["fwd_5d_return"].values
        short_leg = -bot5_gross - ROUND_TRIP_FRICTION  # gross short - 0.0034

        r_period_ls = 0.5 * np.mean(long_leg) + 0.5 * np.mean(short_leg)
        cap_ls *= (1.0 + r_period_ls)
        equity_ls.append(cap_ls)
        period_rets_ls.append(r_period_ls)
        trade_profits_ls.extend(long_leg.tolist())
        trade_profits_ls.extend(short_leg.tolist())

    # Metrics calculation
    # Long-Only
    cagr_lo = (cap_lo ** (1.0 / years) - 1.0) * 100.0 if years > 0 and cap_lo > 0 else -100.0
    peak_lo = np.maximum.accumulate(equity_lo)
    max_dd_lo = np.min((np.array(equity_lo) - peak_lo) / peak_lo) * 100.0 if len(equity_lo) > 0 else 0.0
    win_rate_period_lo = (np.sum(np.array(period_rets_lo) > 0) / len(period_rets_lo) * 100.0) if period_rets_lo else 0.0
    win_rate_trade_lo = (np.sum(np.array(trade_profits_lo) > 0) / len(trade_profits_lo) * 100.0) if trade_profits_lo else 0.0

    # Long-Short
    cagr_ls = (cap_ls ** (1.0 / years) - 1.0) * 100.0 if years > 0 and cap_ls > 0 else -100.0
    peak_ls = np.maximum.accumulate(equity_ls)
    max_dd_ls = np.min((np.array(equity_ls) - peak_ls) / peak_ls) * 100.0 if len(equity_ls) > 0 else 0.0
    win_rate_period_ls = (np.sum(np.array(period_rets_ls) > 0) / len(period_rets_ls) * 100.0) if period_rets_ls else 0.0
    win_rate_trade_ls = (np.sum(np.array(trade_profits_ls) > 0) / len(trade_profits_ls) * 100.0) if trade_profits_ls else 0.0

    return {
        "n_rebalances": len(period_rets_lo),
        "years": years,
        "Long-Only": {
            "CAGR": cagr_lo,
            "MaxDD": max_dd_lo,
            "PeriodWinRate": win_rate_period_lo,
            "TradeWinRate": win_rate_trade_lo,
            "FinalCap": cap_lo,
        },
        "Long-Short": {
            "CAGR": cagr_ls,
            "MaxDD": max_dd_ls,
            "PeriodWinRate": win_rate_period_ls,
            "TradeWinRate": win_rate_trade_ls,
            "FinalCap": cap_ls,
        }
    }


def run_null_simulation(clean_df, start_date="2021-01-01", end_date="2023-12-31", n_sims=1000, seed=42):
    """
    Run 1,000 random 5-stock portfolios rebalanced every 5 days on Period 2 (2021-2023).
    Evaluates null CAGR percentiles under identical cost and STCG tax rules.
    """
    all_dates = sorted(clean_df["trade_date"].unique())
    p2_dates = [d for d in all_dates if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)][::HOLDING_DAYS]
    years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25

    p2_pools = []
    for d in p2_dates:
        day_df = clean_df[clean_df["trade_date"] == d]
        if len(day_df) >= N_STOCKS_SELECT:
            p2_pools.append(day_df["fwd_5d_return"].values)

    rng = np.random.default_rng(seed)
    null_cagrs = np.zeros(n_sims)

    for i in range(n_sims):
        cap = 1.0
        for pool in p2_pools:
            chosen = rng.choice(pool, size=N_STOCKS_SELECT, replace=False)
            net_trade = chosen - ROUND_TRIP_FRICTION
            after_tax = np.where(net_trade > 0, net_trade * (1.0 - STCG_TAX_RATE), net_trade)
            r_period = np.mean(after_tax)
            cap *= (1.0 + r_period)
        cagr = (cap ** (1.0 / years) - 1.0) * 100.0 if cap > 0 else -100.0
        null_cagrs[i] = cagr

    return {
        "50th": np.percentile(null_cagrs, 50),
        "90th": np.percentile(null_cagrs, 90),
        "95th": np.percentile(null_cagrs, 95),
        "99th": np.percentile(null_cagrs, 99),
        "mean": np.mean(null_cagrs),
        "std": np.std(null_cagrs, ddof=1)
    }


def main():
    print("=" * 80)
    print("          DELIVERY PERCENTAGE Z-SCORE: P&L BACKTEST & NULL SIMULATION         ")
    print("=" * 80)
    print(f"Friction / Slippage : {FRICTION_PER_SIDE*100:.2f}% per side ({ROUND_TRIP_FRICTION*100:.2f}% round trip)")
    print(f"STCG Tax (Long-Only): {STCG_TAX_RATE*100:.0f}% on positive trade profits (asymmetric drag)")
    print(f"STCG Tax (L-S)      : None (Futures tax treatment)")
    print(f"Holding Period      : {HOLDING_DAYS} trading days (strictly non-overlapping)")
    print(f"Portfolio Size      : Top {N_STOCKS_SELECT} (Long) / Bottom {N_STOCKS_SELECT} (Short)")

    clean_df = load_clean_data()
    all_dates = sorted(clean_df["trade_date"].unique())
    print(f"Dataset Dates       : {all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')} ({len(all_dates):,} days)")

    # Run Walk-Forward Splits
    results = {}
    for period_name, (start_str, end_str) in PERIODS.items():
        p_dates = [d for d in all_dates if pd.Timestamp(start_str) <= d <= pd.Timestamp(end_str)]
        rebal_dates = p_dates[::HOLDING_DAYS]
        res = simulate_period(clean_df, rebal_dates, start_str, end_str)
        results[period_name] = res

    # Print Summary Table
    print("\n" + "=" * 80)
    print("                      WALK-FORWARD PERFORMANCE SUMMARY                         ")
    print("=" * 80)
    header_fmt = "{:<36} | {:>8} | {:>9} | {:>10} | {:>11} | {:>9}"
    row_fmt    = "{:<36} | {:>7.2f}% | {:>8.2f}% | {:>9.1f}% | {:>10.1f}% | {:>8.4f}x"

    print("\n--- LONG-ONLY STRATEGY (Top 5 Stocks, Friction 0.34%, STCG Tax 15%) ---")
    print(header_fmt.format("Period", "CAGR", "MaxDD", "Period Win", "Trade Win", "Final Cap"))
    print("-" * 92)
    for p_name, res in results.items():
        lo = res["Long-Only"]
        print(row_fmt.format(p_name, lo["CAGR"], lo["MaxDD"], lo["PeriodWinRate"], lo["TradeWinRate"], lo["FinalCap"]))

    print("\n--- LONG-SHORT STRATEGY (Top 5 Long / Bottom 5 Short, Friction 0.34%, No Tax) ---")
    print(header_fmt.format("Period", "CAGR", "MaxDD", "Period Win", "Trade Win", "Final Cap"))
    print("-" * 92)
    for p_name, res in results.items():
        ls = res["Long-Short"]
        print(row_fmt.format(p_name, ls["CAGR"], ls["MaxDD"], ls["PeriodWinRate"], ls["TradeWinRate"], ls["FinalCap"]))

    # Task 4: The Null Test on Period 2 (2021-2023)
    print("\n" + "=" * 80)
    print("           TASK 4: RANDOM NULL TEST (1,000 RUNS, PERIOD 2: 2021-2023)          ")
    print("=" * 80)
    t0 = time.time()
    null_stats = run_null_simulation(clean_df, start_date="2021-01-01", end_date="2023-12-31", n_sims=1000, seed=42)
    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.2f} seconds.")
    print(f"Random 5-Stock Portfolio 50th Percentile (Median) : {null_stats['50th']:>+6.2f}% CAGR")
    print(f"Random 5-Stock Portfolio 90th Percentile          : {null_stats['90th']:>+6.2f}% CAGR")
    print(f"Random 5-Stock Portfolio 95th Percentile          : {null_stats['95th']:>+6.2f}% CAGR")
    print(f"Random 5-Stock Portfolio 99th Percentile          : {null_stats['99th']:>+6.2f}% CAGR")

    true_cagr = results["Period 2 (Out-of-Sample: 2021-2023)"]["Long-Only"]["CAGR"]
    print("\n--- Strategy vs Null Comparison (Period 2: 2021-2023) ---")
    print(f"True Strategy Long-Only CAGR : {true_cagr:>+6.2f}%")
    print(f"Random Null 50th (Median)    : {null_stats['50th']:>+6.2f}%")
    print(f"Random Null 95th Percentile  : {null_stats['95th']:>+6.2f}%")

    if true_cagr > null_stats["95th"]:
        print("[VERDICT] Strategy beats the Random Null 95th percentile in P&L.")
    elif true_cagr > null_stats["50th"]:
        print("[VERDICT] Strategy beats Random Median but fails to reach 95th percentile, and net CAGR is negative.")
    else:
        print("[VERDICT] Strategy fails to beat even the Random Median after friction and taxes.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
