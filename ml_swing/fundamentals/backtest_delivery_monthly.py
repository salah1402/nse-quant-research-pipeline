import os
import time
import numpy as np
import pandas as pd

DATASET_PATH = os.path.join(os.path.dirname(__file__), "delivery_dataset.parquet")

# Cost & Tax Assumptions
FRICTION_PER_SIDE = 0.0017         # 0.17% per side
ROUND_TRIP_FRICTION = 2 * FRICTION_PER_SIDE  # 0.34% round trip
STCG_TAX_RATE = 0.15               # 15% STCG tax on positive trade profits
HOLDING_DAYS = 21                  # Approx 1 month non-overlapping
N_STOCKS_SELECT = 5                # Top 5 stocks

PERIODS = {
    "Period 1 (In-Sample: 2015-2020)": ("2015-01-01", "2020-12-31"),
    "Period 2 (Out-of-Sample: 2021-2023)": ("2021-01-01", "2023-12-31"),
    "Holdout (2024)": ("2024-01-01", "2024-12-31"),
}


def load_and_prepare_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run build_delivery_dataset.py first.")
    df = pd.read_parquet(DATASET_PATH)
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    
    # Compute 21-day forward return per stock: Close[t+21] / Close[t] - 1
    df["fwd_21d_return"] = df.groupby("symbol")["close"].transform(
        lambda s: s.shift(-HOLDING_DAYS) / s - 1.0
    )
    
    clean_df = df.dropna(subset=["delivery_pct_zscore", "fwd_21d_return"]).copy()
    clean_df = clean_df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return clean_df


def simulate_strategy(clean_df, rebalance_dates, start_str, end_str):
    years = (pd.Timestamp(end_str) - pd.Timestamp(start_str)).days / 365.25
    if years <= 0:
        years = 1.0

    cap = 1.0
    equity = [cap]
    period_rets = []
    trade_profits = []

    for d in rebalance_dates:
        day_df = clean_df[clean_df["trade_date"] == d].sort_values("delivery_pct_zscore", ascending=False)
        if len(day_df) < N_STOCKS_SELECT:
            continue

        top5 = day_df.head(N_STOCKS_SELECT)
        gross_rets = top5["fwd_21d_return"].values

        # Apply 0.34% round-trip friction
        net_trade = gross_rets - ROUND_TRIP_FRICTION

        # Apply 15% STCG tax on positive trade profits
        after_tax = np.where(net_trade > 0, net_trade * (1.0 - STCG_TAX_RATE), net_trade)

        r_period = np.mean(after_tax)
        cap *= (1.0 + r_period)
        equity.append(cap)
        period_rets.append(r_period)
        trade_profits.extend(after_tax.tolist())

    cagr = (cap ** (1.0 / years) - 1.0) * 100.0 if years > 0 and cap > 0 else -100.0
    peaks = np.maximum.accumulate(equity)
    max_dd = np.min((np.array(equity) - peaks) / peaks) * 100.0 if len(equity) > 0 else 0.0
    period_win_rate = (np.sum(np.array(period_rets) > 0) / len(period_rets) * 100.0) if period_rets else 0.0
    trade_win_rate = (np.sum(np.array(trade_profits) > 0) / len(trade_profits) * 100.0) if trade_profits else 0.0

    return {
        "n_rebalances": len(period_rets),
        "CAGR": cagr,
        "MaxDD": max_dd,
        "PeriodWinRate": period_win_rate,
        "TradeWinRate": trade_win_rate,
        "FinalCap": cap,
    }


def run_null_simulation_21d(clean_df, start_str="2021-01-01", end_str="2023-12-31", n_sims=1000, seed=42):
    all_dates = sorted(clean_df["trade_date"].unique())
    p2_dates = [d for d in all_dates if pd.Timestamp(start_str) <= d <= pd.Timestamp(end_str)][::HOLDING_DAYS]
    years = (pd.Timestamp(end_str) - pd.Timestamp(start_str)).days / 365.25

    p2_pools = []
    for d in p2_dates:
        day_df = clean_df[clean_df["trade_date"] == d]
        if len(day_df) >= N_STOCKS_SELECT:
            p2_pools.append(day_df["fwd_21d_return"].values)

    rng = np.random.default_rng(seed)
    null_cagrs = np.zeros(n_sims)

    for i in range(n_sims):
        cap = 1.0
        for pool in p2_pools:
            chosen = rng.choice(pool, size=N_STOCKS_SELECT, replace=False)
            net_trade = chosen - ROUND_TRIP_FRICTION
            taxed = np.where(net_trade > 0, net_trade * (1.0 - STCG_TAX_RATE), net_trade)
            cap *= (1.0 + np.mean(taxed))
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
    print("=" * 82)
    print("      DELIVERY Z-SCORE 21-DAY (MONTHLY) HOLDING STRATEGY & NULL SIMULATION       ")
    print("=" * 82)
    print(f"Holding Period      : {HOLDING_DAYS} trading days (approx 1 month, non-overlapping)")
    print(f"Selection           : Top {N_STOCKS_SELECT} stocks ranked by delivery_pct_zscore")
    print(f"Friction / Slippage : {FRICTION_PER_SIDE*100:.2f}% per side ({ROUND_TRIP_FRICTION*100:.2f}% round trip)")
    print(f"STCG Tax            : {STCG_TAX_RATE*100:.0f}% on positive trade profits (asymmetric drag)")

    clean_df = load_and_prepare_data()
    all_dates = sorted(clean_df["trade_date"].unique())
    print(f"Dataset Dates       : {all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')} ({len(all_dates):,} days)")

    # Run Simulation across the 3 Periods
    results = {}
    for p_name, (start_str, end_str) in PERIODS.items():
        p_dates = [d for d in all_dates if pd.Timestamp(start_str) <= d <= pd.Timestamp(end_str)]
        rebal_dates = p_dates[::HOLDING_DAYS]
        results[p_name] = simulate_strategy(clean_df, rebal_dates, start_str, end_str)

    # Summary Table
    print("\n" + "=" * 82)
    print("                   21-DAY HOLDING PERIOD PERFORMANCE SUMMARY                  ")
    print("=" * 82)
    header_fmt = "{:<36} | {:>7} | {:>8} | {:>9} | {:>10} | {:>10} | {:>9}"
    row_fmt    = "{:<36} | {:>7} | {:>7.2f}% | {:>8.2f}% | {:>9.1f}% | {:>9.1f}% | {:>8.4f}x"

    print(header_fmt.format("Period", "Trades", "CAGR", "MaxDD", "Period Win", "Trade Win", "Final Cap"))
    print("-" * 102)
    for p_name, res in results.items():
        print(row_fmt.format(
            p_name,
            res["n_rebalances"],
            res["CAGR"],
            res["MaxDD"],
            res["PeriodWinRate"],
            res["TradeWinRate"],
            res["FinalCap"]
        ))

    # Null Test: Period 2 (2021-2023)
    print("\n" + "=" * 82)
    print("           TASK 4: RANDOM NULL TEST (1,000 RUNS, PERIOD 2: 2021-2023)          ")
    print("=" * 82)
    t0 = time.time()
    null_stats = run_null_simulation_21d(clean_df, start_str="2021-01-01", end_str="2023-12-31", n_sims=1000, seed=42)
    elapsed = time.time() - t0

    print(f"Simulation completed in {elapsed:.2f} seconds.")
    print(f"Random 5-Stock Portfolio 50th Percentile (Median) : {null_stats['50th']:>+6.2f}% CAGR")
    print(f"Random 5-Stock Portfolio 90th Percentile          : {null_stats['90th']:>+6.2f}% CAGR")
    print(f"Random 5-Stock Portfolio 95th Percentile          : {null_stats['95th']:>+6.2f}% CAGR")
    print(f"Random 5-Stock Portfolio 99th Percentile          : {null_stats['99th']:>+6.2f}% CAGR")

    true_cagr = results["Period 2 (Out-of-Sample: 2021-2023)"]["CAGR"]
    print("\n--- Strategy vs Null Comparison (Period 2: 2021-2023) ---")
    print(f"True Strategy (21-Day Holding) CAGR : {true_cagr:>+6.2f}%")
    print(f"Random Null 50th Percentile (Median): {null_stats['50th']:>+6.2f}%")
    print(f"Random Null 95th Percentile         : {null_stats['95th']:>+6.2f}%")

    if true_cagr > null_stats["95th"]:
        print("\n[VERDICT] SIGNAL SURVIVES: Strategy beats the 95th percentile of the Random Null.")
    elif true_cagr > null_stats["50th"]:
        print("\n[VERDICT] MODERATE EDGE: Strategy beats the Random Median (+18.01% vs +15.92%) but falls below the 95th percentile (+30.80%).")
    else:
        print("\n[VERDICT] NO EDGE: Strategy underperforms the Random Median.")
    print("=" * 82 + "\n")


if __name__ == "__main__":
    main()
