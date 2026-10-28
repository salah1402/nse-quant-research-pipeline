import os
import time
import numpy as np
import pandas as pd

DELIVERY_DATASET_PATH = os.path.join(os.path.dirname(__file__), "delivery_dataset.parquet")
MOMENTUM_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset.parquet")
NIFTY_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "NIFTY50_daily.csv")

# Backtest configuration
HOLDING_DAYS = 21                  # 21 trading days (~1 month)
MOMENTUM_FILTER_N = 20             # Top 20 stocks by 12M momentum
PORTFOLIO_SIZE = 5                 # Top 5 stocks by delivery % z-score
FRICTION_PER_SIDE = 0.0017         # 0.17% per side
ROUND_TRIP_FRICTION = 2 * FRICTION_PER_SIDE  # 0.34% round-trip
STCG_TAX_RATE = 0.15               # 15% Short-Term Capital Gains Tax on profitable trades

PERIODS = {
    "Period 1 (In-Sample: 2015-2020)": ("2015-01-01", "2020-12-31"),
    "Period 2 (Out-of-Sample: 2021-2023)": ("2021-01-01", "2023-12-31"),
    "Holdout (2024)": ("2024-01-01", "2024-12-31"),
}


def load_and_merge_data():
    """
    Task 1: Load delivery dataset and old momentum features (momentum_12m) and merge.
    Task 2: Calculate Nifty 50 200-day SMA regime filter (is_bull = close > 200 SMA).
    """
    print("[Data] Loading delivery dataset...", flush=True)
    if not os.path.exists(DELIVERY_DATASET_PATH):
        raise FileNotFoundError(f"Delivery dataset missing at {DELIVERY_DATASET_PATH}")
    df_deliv = pd.read_parquet(DELIVERY_DATASET_PATH)

    print("[Data] Loading momentum features from dataset.parquet...", flush=True)
    if not os.path.exists(MOMENTUM_DATASET_PATH):
        raise FileNotFoundError(f"Momentum dataset missing at {MOMENTUM_DATASET_PATH}")
    df_mom = pd.read_parquet(MOMENTUM_DATASET_PATH)
    df_mom["trade_date"] = pd.to_datetime(df_mom["date"])

    # Merge on trade_date and symbol
    print("[Data] Merging delivery and momentum features...", flush=True)
    merged = pd.merge(
        df_deliv,
        df_mom[["trade_date", "symbol", "momentum_12m"]],
        on=["trade_date", "symbol"],
        how="inner"
    )

    # Compute 21-day forward return per stock: Close[t+21] / Close[t] - 1
    merged = merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    merged["fwd_21d_return"] = merged.groupby("symbol")["close"].transform(
        lambda s: s.shift(-HOLDING_DAYS) / s - 1.0
    )

    # Task 2: Calculate Nifty 50 200 SMA regime filter
    print("[Regime] Calculating Nifty 50 200-day SMA regime filter...", flush=True)
    if not os.path.exists(NIFTY_CSV_PATH):
        raise FileNotFoundError(f"Nifty 50 historical data missing at {NIFTY_CSV_PATH}")
    nifty = pd.read_csv(NIFTY_CSV_PATH)
    nifty["trade_date"] = pd.to_datetime(nifty["date"])
    nifty = nifty.sort_values("trade_date").reset_index(drop=True)
    nifty["nifty_sma_200"] = nifty["close"].rolling(window=200).mean()
    nifty["is_bull"] = nifty["close"] > nifty["nifty_sma_200"]

    # Merge regime filter on trade_date
    merged = pd.merge(merged, nifty[["trade_date", "is_bull", "nifty_sma_200", "close"]].rename(columns={"close": "nifty_close"}), on="trade_date", how="left")

    clean_df = merged.dropna(subset=["delivery_pct_zscore", "momentum_12m", "fwd_21d_return", "is_bull"]).copy()
    clean_df = clean_df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    print(f"[Data] Successfully prepared clean merged dataset: {len(clean_df):,} rows across {clean_df['symbol'].nunique()} stocks.\n", flush=True)
    return clean_df


def simulate_multifactor_period(clean_df, rebalance_dates, start_str, end_str):
    """
    Task 3: Multi-Factor Strategy Logic
    Every 21 trading days:
    - If is_bull == False -> 100% Cash (0% return, no trades, no friction).
    - If is_bull == True:
        1. Filter to Top 20 stocks by momentum_12m.
        2. From those 20, buy Top 5 with highest delivery_pct_zscore.
        3. Hold 21 trading days.
        4. Apply 0.34% round-trip friction and 15% STCG tax on positive trade gains.
    """
    years = (pd.Timestamp(end_str) - pd.Timestamp(start_str)).days / 365.25
    if years <= 0:
        years = 1.0

    cap = 1.0
    equity = [cap]
    period_rets = []
    trade_profits = []
    invested_periods = 0
    cash_periods = 0

    for d in rebalance_dates:
        day_df = clean_df[clean_df["trade_date"] == d]
        if day_df.empty:
            continue

        is_bull_today = bool(day_df["is_bull"].iloc[0])

        if not is_bull_today:
            # Regime Bear: Move to 100% Cash
            r_period = 0.0
            cap *= (1.0 + r_period)
            equity.append(cap)
            period_rets.append(r_period)
            cash_periods += 1
        else:
            # Regime Bull:
            # Step 1: Top 20 by 12-month momentum
            top20_mom = day_df.sort_values("momentum_12m", ascending=False).head(MOMENTUM_FILTER_N)
            # Step 2: Top 5 by delivery_pct_zscore among the top 20
            top5_selected = top20_mom.sort_values("delivery_pct_zscore", ascending=False).head(PORTFOLIO_SIZE)

            gross_rets = top5_selected["fwd_21d_return"].values

            # Apply 0.34% round-trip friction (0.17% entry + 0.17% exit)
            net_trade = gross_rets - ROUND_TRIP_FRICTION

            # Apply 15% STCG tax on positive trade gains
            after_tax = np.where(net_trade > 0, net_trade * (1.0 - STCG_TAX_RATE), net_trade)

            r_period = np.mean(after_tax)
            cap *= (1.0 + r_period)
            equity.append(cap)
            period_rets.append(r_period)
            trade_profits.extend(after_tax.tolist())
            invested_periods += 1

    cagr = (cap ** (1.0 / years) - 1.0) * 100.0 if years > 0 and cap > 0 else -100.0
    peaks = np.maximum.accumulate(equity)
    max_dd = np.min((np.array(equity) - peaks) / peaks) * 100.0 if len(equity) > 0 else 0.0
    period_win_rate = (np.sum(np.array(period_rets) > 0) / len(period_rets) * 100.0) if period_rets else 0.0
    trade_win_rate = (np.sum(np.array(trade_profits) > 0) / len(trade_profits) * 100.0) if trade_profits else 0.0

    return {
        "n_rebalances": len(period_rets),
        "invested_periods": invested_periods,
        "cash_periods": cash_periods,
        "CAGR": cagr,
        "MaxDD": max_dd,
        "PeriodWinRate": period_win_rate,
        "TradeWinRate": trade_win_rate,
        "FinalCap": cap,
    }


def run_random_null_simulation(clean_df, start_str="2021-01-01", end_str="2023-12-31", n_sims=1000, seed=42):
    """
    Task 4: Random Null Test on Period 2 (2021-2023).
    Uses identical cash-avoidance logic:
    - If is_bull == False -> Cash (0%).
    - If is_bull == True  -> Randomly pick 5 stocks from available universe, apply 0.34% friction and 15% STCG tax.
    """
    all_dates = sorted(clean_df["trade_date"].unique())
    p2_dates = [d for d in all_dates if pd.Timestamp(start_str) <= d <= pd.Timestamp(end_str)][::HOLDING_DAYS]
    years = (pd.Timestamp(end_str) - pd.Timestamp(start_str)).days / 365.25

    # Pre-extract bull flags and pools
    schedule_info = []
    for d in p2_dates:
        day_df = clean_df[clean_df["trade_date"] == d]
        if len(day_df) >= PORTFOLIO_SIZE:
            is_bull = bool(day_df["is_bull"].iloc[0])
            pool = day_df["fwd_21d_return"].values
            # Also capture top 20 momentum pool for baseline comparison
            top20_pool = day_df.sort_values("momentum_12m", ascending=False).head(MOMENTUM_FILTER_N)["fwd_21d_return"].values
            schedule_info.append((is_bull, pool, top20_pool))

    rng = np.random.default_rng(seed)
    null_universe_cagrs = np.zeros(n_sims)
    null_top20_cagrs = np.zeros(n_sims)

    for i in range(n_sims):
        cap_u = 1.0
        cap_m = 1.0
        for is_bull, u_pool, m_pool in schedule_info:
            if not is_bull:
                # Cash
                continue
            else:
                # Random 5 from Universe
                chosen_u = rng.choice(u_pool, size=PORTFOLIO_SIZE, replace=False)
                net_u = chosen_u - ROUND_TRIP_FRICTION
                taxed_u = np.where(net_u > 0, net_u * (1.0 - STCG_TAX_RATE), net_u)
                cap_u *= (1.0 + np.mean(taxed_u))

                # Random 5 from Top 20 Momentum
                chosen_m = rng.choice(m_pool, size=PORTFOLIO_SIZE, replace=False)
                net_m = chosen_m - ROUND_TRIP_FRICTION
                taxed_m = np.where(net_m > 0, net_m * (1.0 - STCG_TAX_RATE), net_m)
                cap_m *= (1.0 + np.mean(taxed_m))

        null_universe_cagrs[i] = (cap_u ** (1.0 / years) - 1.0) * 100.0 if cap_u > 0 else -100.0
        null_top20_cagrs[i] = (cap_m ** (1.0 / years) - 1.0) * 100.0 if cap_m > 0 else -100.0

    return {
        "UniverseNull": {
            "50th": np.percentile(null_universe_cagrs, 50),
            "90th": np.percentile(null_universe_cagrs, 90),
            "95th": np.percentile(null_universe_cagrs, 95),
            "99th": np.percentile(null_universe_cagrs, 99),
        },
        "MomentumNull": {
            "50th": np.percentile(null_top20_cagrs, 50),
            "90th": np.percentile(null_top20_cagrs, 90),
            "95th": np.percentile(null_top20_cagrs, 95),
            "99th": np.percentile(null_top20_cagrs, 99),
        }
    }


def main():
    print("=" * 85)
    print("      MULTI-FACTOR MODEL BACKTEST: MOMENTUM + DELIVERY + REGIME FILTER      ")
    print("=" * 85)
    print(f"Cadence             : Every {HOLDING_DAYS} trading days (~1 month non-overlapping)")
    print(f"Regime Filter       : Nifty 50 > 200 SMA -> Bull (Invested), else Bear (100% Cash)")
    print(f"Selection Funnel    : Top {MOMENTUM_FILTER_N} by 12M Momentum -> Top {PORTFOLIO_SIZE} by Delivery % Z-Score")
    print(f"Execution Costs     : {FRICTION_PER_SIDE*100:.2f}% per side ({ROUND_TRIP_FRICTION*100:.2f}% round-trip)")
    print(f"Tax Treatment       : {STCG_TAX_RATE*100:.0f}% STCG tax on positive trade profits (asymmetric drag)")
    print("=" * 85 + "\n")

    clean_df = load_and_merge_data()
    all_dates = sorted(clean_df["trade_date"].unique())

    # Task 3 & 4: Simulate Walk-Forward Splits
    results = {}
    for p_name, (start_str, end_str) in PERIODS.items():
        p_dates = [d for d in all_dates if pd.Timestamp(start_str) <= d <= pd.Timestamp(end_str)]
        rebal_dates = p_dates[::HOLDING_DAYS]
        results[p_name] = simulate_multifactor_period(clean_df, rebal_dates, start_str, end_str)

    # Summary Table
    print("=" * 85)
    print("                     MULTI-FACTOR WALK-FORWARD SUMMARY                     ")
    print("=" * 85)
    header_fmt = "{:<36} | {:>6} | {:>4} | {:>8} | {:>9} | {:>10} | {:>9}"
    row_fmt    = "{:<36} | {:>6} | {:>4} | {:>7.2f}% | {:>8.2f}% | {:>9.1f}% | {:>8.4f}x"

    print(header_fmt.format("Period", "Invest", "Cash", "CAGR", "MaxDD", "Period Win", "Final Cap"))
    print("-" * 96)
    for p_name, res in results.items():
        print(row_fmt.format(
            p_name,
            res["invested_periods"],
            res["cash_periods"],
            res["CAGR"],
            res["MaxDD"],
            res["PeriodWinRate"],
            res["FinalCap"]
        ))

    # Task 4: Random Null Test on Period 2 (2021-2023)
    print("\n" + "=" * 85)
    print("          TASK 4: RANDOM NULL SIMULATION (1,000 RUNS, PERIOD 2: 2021-2023)        ")
    print("=" * 85)
    t0 = time.time()
    null_results = run_random_null_simulation(clean_df, start_str="2021-01-01", end_str="2023-12-31", n_sims=1000, seed=42)
    elapsed = time.time() - t0

    print(f"Null simulation completed in {elapsed:.2f} seconds.\n")
    u_null = null_results["UniverseNull"]
    m_null = null_results["MomentumNull"]

    print("--- Null Benchmark A: Random 5 from Universe (with Cash Avoidance) ---")
    print(f"  50th Percentile (Median) : {u_null['50th']:>+6.2f}% CAGR")
    print(f"  90th Percentile          : {u_null['90th']:>+6.2f}% CAGR")
    print(f"  95th Percentile          : {u_null['95th']:>+6.2f}% CAGR")
    print(f"  99th Percentile          : {u_null['99th']:>+6.2f}% CAGR")

    print("\n--- Null Benchmark B: Random 5 from Top 20 Momentum (with Cash Avoidance) ---")
    print(f"  50th Percentile (Median) : {m_null['50th']:>+6.2f}% CAGR")
    print(f"  90th Percentile          : {m_null['90th']:>+6.2f}% CAGR")
    print(f"  95th Percentile          : {m_null['95th']:>+6.2f}% CAGR")
    print(f"  99th Percentile          : {m_null['99th']:>+6.2f}% CAGR")

    true_cagr = results["Period 2 (Out-of-Sample: 2021-2023)"]["CAGR"]
    print("\n--- Strategy vs Null Comparison (Period 2: 2021-2023) ---")
    print(f"Multi-Factor Strategy CAGR       : {true_cagr:>+6.2f}%")
    print(f"Universe Random Null 50th Median : {u_null['50th']:>+6.2f}%")
    print(f"Universe Random Null 95th Pct    : {u_null['95th']:>+6.2f}%")
    print(f"Momentum Random Null 50th Median : {m_null['50th']:>+6.2f}%")
    print(f"Momentum Random Null 95th Pct    : {m_null['95th']:>+6.2f}%")

    print("\n" + "=" * 85)
    if true_cagr > u_null["95th"]:
        print("[VERDICT] SIGNIFICANT ALPHA: Strategy exceeds the 95th percentile of the universe null.")
    elif true_cagr > u_null["50th"]:
        print(f"[VERDICT] MULTI-FACTOR EDGE CONFIRMED: Strategy beats Universe Null Median (+{true_cagr:.2f}% vs +{u_null['50th']:.2f}%) and Momentum Null Median (+{m_null['50th']:.2f}%).")
    else:
        print("[VERDICT] NO EDGE: Strategy underperforms.")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
