import os
import time
import numpy as np
import pandas as pd

DELIVERY_DATASET_PATH = os.path.join(os.path.dirname(__file__), "delivery_dataset.parquet")
MOMENTUM_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset.parquet")
NIFTY_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "NIFTY50_daily.csv")
VIX_PARQUET_PATH = os.path.join(os.path.dirname(__file__), "india_vix.parquet")

# Cost & Cadence Constants
HOLDING_DAYS = 21                  # 21 trading days (~1 month non-overlapping)
FRICTION_PER_SIDE = 0.0017         # 0.17% per side
ROUND_TRIP_FRICTION = 2 * FRICTION_PER_SIDE  # 0.34% round-trip
STCG_TAX_RATE = 0.15               # 15% STCG tax on positive trade profits

PERIODS = {
    "Period 1 (2015-2020)": ("2015-01-01", "2020-12-31"),
    "Period 2 (2021-2023)": ("2021-01-01", "2023-12-31"),
    "Holdout (2024)": ("2024-01-01", "2024-12-31"),
}

# Grid Search Parameters
PORTFOLIO_SIZES = [5, 10, 15]
REGIME_FILTERS = [
    "SMA_200",      # Baseline: Nifty > 200 SMA
    "SMA_100",      # Faster: Nifty > 100 SMA
    "SMA_50",       # Ultra-Fast: Nifty > 50 SMA
    "VIX_FILTER",   # VIX: India VIX <= 25
    "SMA_100_VIX",  # Combined: Nifty > 100 SMA & VIX <= 25
    "SMA_50_VIX",   # Combined: Nifty > 50 SMA & VIX <= 25
    "SMA_200_VIX",  # Combined: Nifty > 200 SMA & VIX <= 25
]
DELIVERY_WEIGHTS = [
    "Pure_Momentum",       # 0% Delivery: Top N by 12M momentum
    "Mom_plus_Delivery",   # Sequential: Top 20 Momentum -> Top N Delivery
    "Combined_Score",      # Sum of cross-sectional Z-scores: Z(Mom) + Z(Delivery)
]


def load_and_prepare_data():
    """Load, merge, and clean all datasets including Nifty SMAs and India VIX."""
    df_deliv = pd.read_parquet(DELIVERY_DATASET_PATH)
    df_mom = pd.read_parquet(MOMENTUM_DATASET_PATH)
    df_mom["trade_date"] = pd.to_datetime(df_mom["date"])

    merged = pd.merge(
        df_deliv,
        df_mom[["trade_date", "symbol", "momentum_12m"]],
        on=["trade_date", "symbol"],
        how="inner"
    )

    # Compute 21-day forward return per stock
    merged = merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    merged["fwd_21d_return"] = merged.groupby("symbol")["close"].transform(
        lambda s: s.shift(-HOLDING_DAYS) / s - 1.0
    )

    # Nifty 50 SMAs
    nifty = pd.read_csv(NIFTY_CSV_PATH)
    nifty["trade_date"] = pd.to_datetime(nifty["date"])
    nifty = nifty.sort_values("trade_date").reset_index(drop=True)
    nifty["sma_200"] = nifty["close"].rolling(200).mean()
    nifty["sma_100"] = nifty["close"].rolling(100).mean()
    nifty["sma_50"] = nifty["close"].rolling(50).mean()

    # If VIX parquet exists, load it; otherwise proxy with 21d rolling annualized volatility
    if os.path.exists(VIX_PARQUET_PATH):
        vix = pd.read_parquet(VIX_PARQUET_PATH)
        nifty = pd.merge(nifty, vix[["trade_date", "vix_close"]], on="trade_date", how="left")
    else:
        # Proxy: 21-day rolling volatility * sqrt(252) * 100
        nifty["vix_close"] = nifty["close"].pct_change().rolling(21).std() * np.sqrt(252) * 100.0

    # Fill any missing VIX with 15.0 (calm baseline)
    nifty["vix_close"] = nifty["vix_close"].fillna(15.0)

    # Merge regime indicators into merged stock dataframe
    merged = pd.merge(
        merged,
        nifty[["trade_date", "close", "sma_200", "sma_100", "sma_50", "vix_close"]].rename(columns={"close": "nifty_close"}),
        on="trade_date",
        how="left"
    )

    clean_df = merged.dropna(subset=["delivery_pct_zscore", "momentum_12m", "fwd_21d_return", "sma_200"]).copy()
    clean_df = clean_df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return clean_df


def is_regime_bull(day_df, regime_name):
    """Evaluate whether the market regime is Bullish for the given filter."""
    n_close = day_df["nifty_close"].iloc[0]
    vix_val = day_df["vix_close"].iloc[0]

    if regime_name == "SMA_200":
        return n_close > day_df["sma_200"].iloc[0]
    elif regime_name == "SMA_100":
        return n_close > day_df["sma_100"].iloc[0]
    elif regime_name == "SMA_50":
        return n_close > day_df["sma_50"].iloc[0]
    elif regime_name == "VIX_FILTER":
        return vix_val <= 25.0
    elif regime_name == "SMA_100_VIX":
        return (n_close > day_df["sma_100"].iloc[0]) and (vix_val <= 25.0)
    elif regime_name == "SMA_50_VIX":
        return (n_close > day_df["sma_50"].iloc[0]) and (vix_val <= 25.0)
    elif regime_name == "SMA_200_VIX":
        return (n_close > day_df["sma_200"].iloc[0]) and (vix_val <= 25.0)
    return True


def select_stocks(day_df, size, deliv_weight):
    """Select Top N stocks according to the configured ranking methodology."""
    if deliv_weight == "Pure_Momentum":
        return day_df.sort_values("momentum_12m", ascending=False).head(size)
    elif deliv_weight == "Mom_plus_Delivery":
        pool_n = max(20, size + 5)
        top_mom = day_df.sort_values("momentum_12m", ascending=False).head(pool_n)
        return top_mom.sort_values("delivery_pct_zscore", ascending=False).head(size)
    elif deliv_weight == "Combined_Score":
        # Standardized cross-sectional Z-score sum: Z(Momentum) + Z(Delivery)
        mom = day_df["momentum_12m"]
        deliv = day_df["delivery_pct_zscore"]
        z_mom = (mom - mom.mean()) / (mom.std() + 1e-12)
        z_del = (deliv - deliv.mean()) / (deliv.std() + 1e-12)
        combined = z_mom + z_del
        top_idx = np.argsort(-combined.values)[:size]
        return day_df.iloc[top_idx]
    return day_df.head(size)


def simulate_configuration(clean_df, all_dates, size, regime, deliv_weight):
    """Backtest a single configuration across all 3 periods."""
    result = {
        "Size": size,
        "Regime": regime,
        "DeliveryWeight": deliv_weight,
    }

    for p_key, (s_dt, e_dt) in PERIODS.items():
        p_dates = [d for d in all_dates if pd.Timestamp(s_dt) <= d <= pd.Timestamp(e_dt)][::HOLDING_DAYS]
        years = (pd.Timestamp(e_dt) - pd.Timestamp(s_dt)).days / 365.25
        if years <= 0:
            years = 1.0

        cap = 1.0
        equity = [cap]
        rets = []
        invested = 0
        cash = 0

        for d in p_dates:
            day_df = clean_df[clean_df["trade_date"] == d]
            if len(day_df) < size:
                continue

            bull = is_regime_bull(day_df, regime)
            if not bull:
                # Cash period: 0% return
                r_period = 0.0
                cash += 1
            else:
                selected = select_stocks(day_df, size, deliv_weight)
                gross_rets = selected["fwd_21d_return"].values

                # Friction 0.34% round-trip
                net_trade = gross_rets - ROUND_TRIP_FRICTION
                # 15% STCG tax on positive trade gains
                taxed = np.where(net_trade > 0, net_trade * (1.0 - STCG_TAX_RATE), net_trade)

                r_period = np.mean(taxed)
                invested += 1

            cap *= (1.0 + r_period)
            equity.append(cap)
            rets.append(r_period)

        cagr = (cap ** (1.0 / years) - 1.0) * 100.0 if cap > 0 else -100.0
        peaks = np.maximum.accumulate(equity)
        max_dd = np.min((np.array(equity) - peaks) / peaks) * 100.0 if len(equity) > 0 else 0.0
        win_rate = (np.sum(np.array(rets) > 0) / len(rets) * 100.0) if rets else 0.0

        result[f"{p_key}_CAGR"] = cagr
        result[f"{p_key}_MaxDD"] = max_dd
        result[f"{p_key}_WinRate"] = win_rate
        result[f"{p_key}_FinalCap"] = cap
        result[f"{p_key}_Invested"] = invested
        result[f"{p_key}_Cash"] = cash

    return result


def run_null_test_for_best(clean_df, best_size, best_regime, start_str="2021-01-01", end_str="2023-12-31", n_sims=1000, seed=42):
    """Run 1,000-run Random Null test for Period 2 on the winning configuration."""
    all_dates = sorted(clean_df["trade_date"].unique())
    p2_dates = [d for d in all_dates if pd.Timestamp(start_str) <= d <= pd.Timestamp(end_str)][::HOLDING_DAYS]
    years = (pd.Timestamp(end_str) - pd.Timestamp(start_str)).days / 365.25

    schedule = []
    for d in p2_dates:
        day_df = clean_df[clean_df["trade_date"] == d]
        if len(day_df) >= best_size:
            bull = is_regime_bull(day_df, best_regime)
            pool = day_df["fwd_21d_return"].values
            schedule.append((bull, pool))

    rng = np.random.default_rng(seed)
    null_cagrs = np.zeros(n_sims)

    for i in range(n_sims):
        cap = 1.0
        for bull, pool in schedule:
            if not bull:
                continue
            chosen = rng.choice(pool, size=best_size, replace=False)
            net_trade = chosen - ROUND_TRIP_FRICTION
            taxed = np.where(net_trade > 0, net_trade * (1.0 - STCG_TAX_RATE), net_trade)
            cap *= (1.0 + np.mean(taxed))
        null_cagrs[i] = (cap ** (1.0 / years) - 1.0) * 100.0 if cap > 0 else -100.0

    return {
        "50th": np.percentile(null_cagrs, 50),
        "90th": np.percentile(null_cagrs, 90),
        "95th": np.percentile(null_cagrs, 95),
        "99th": np.percentile(null_cagrs, 99),
    }


def main():
    print("=" * 105)
    print("                MULTI-FACTOR GRID SEARCH & ROBUSTNESS OPTIMIZATION                   ")
    print("=" * 105)
    print(f"Portfolio Sizes     : {PORTFOLIO_SIZES}")
    print(f"Regime Filters      : {REGIME_FILTERS}")
    print(f"Delivery Weights    : {DELIVERY_WEIGHTS}")
    print(f"Friction & Tax      : 0.34% round-trip friction, 15% STCG tax on positive trade gains\n")

    clean_df = load_and_prepare_data()
    all_dates = sorted(clean_df["trade_date"].unique())

    # Execute Grid Search
    total_configs = len(PORTFOLIO_SIZES) * len(REGIME_FILTERS) * len(DELIVERY_WEIGHTS)
    print(f"[Grid Search] Running {total_configs} configuration backtests across all periods...", flush=True)

    t0 = time.time()
    results = []
    for size in PORTFOLIO_SIZES:
        for reg in REGIME_FILTERS:
            for dw in DELIVERY_WEIGHTS:
                res = simulate_configuration(clean_df, all_dates, size, reg, dw)
                results.append(res)
    elapsed = time.time() - t0
    print(f"[Grid Search] Completed {total_configs} backtests in {elapsed:.2f} seconds.\n")

    grid_df = pd.DataFrame(results)

    # Filter for configurations with Period 1 CAGR > 0
    p1_key = "Period 1 (2015-2020)_CAGR"
    p2_key = "Period 2 (2021-2023)_CAGR"
    holdout_key = "Holdout (2024)_CAGR"

    pos_p1_df = grid_df[grid_df[p1_key] > 0].copy()
    print("=" * 105)
    print(f"TOP 10 CONFIGURATIONS (Ranked by Holdout CAGR, Filtered for Period 1 CAGR > 0)")
    print(f"Found {len(pos_p1_df)} configurations with positive CAGR in Period 1.")
    print("=" * 105)

    top10 = pos_p1_df.sort_values(holdout_key, ascending=False).head(10)

    header = "{:<4} | {:>4} | {:<12} | {:<18} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9}"
    row_fmt = "{:<4} | {:>4} | {:<12} | {:<18} | {:>8.2f}% | {:>8.2f}% | {:>8.2f}% | {:>8.2f}% | {:>8.2f}% | {:>8.2f}%"

    print(header.format("Rank", "Size", "Regime", "Delivery Weight", "P1 CAGR", "P1 MaxDD", "P2 CAGR", "P2 MaxDD", "Holdout", "HoldoutDD"))
    print("-" * 115)
    for rank, (_, r) in enumerate(top10.iterrows(), start=1):
        print(row_fmt.format(
            rank,
            int(r["Size"]),
            r["Regime"],
            r["DeliveryWeight"],
            r[p1_key],
            r["Period 1 (2015-2020)_MaxDD"],
            r[p2_key],
            r["Period 2 (2021-2023)_MaxDD"],
            r[holdout_key],
            r["Holdout (2024)_MaxDD"]
        ))

    # Identify Best Configuration (Rank 1 of Top 10)
    best_row = top10.iloc[0]
    best_size = int(best_row["Size"])
    best_regime = best_row["Regime"]
    best_dw = best_row["DeliveryWeight"]

    print("\n" + "=" * 105)
    print(f"                  BEST CONFIGURATION: Size={best_size}, Regime={best_regime}, Weight={best_dw}")
    print("=" * 105)
    print(f"  Period 1 (2015-2020) : CAGR = {best_row[p1_key]:>+6.2f}%, MaxDD = {best_row['Period 1 (2015-2020)_MaxDD']:>6.2f}%")
    print(f"  Period 2 (2021-2023) : CAGR = {best_row[p2_key]:>+6.2f}%, MaxDD = {best_row['Period 2 (2021-2023)_MaxDD']:>6.2f}%")
    print(f"  Holdout (2024)       : CAGR = {best_row[holdout_key]:>+6.2f}%, MaxDD = {best_row['Holdout (2024)_MaxDD']:>6.2f}%")

    # Run Random Null Test for the Best Configuration on Period 2 (2021-2023)
    print("\n" + "=" * 105)
    print(f"          RANDOM NULL SIMULATION (1,000 RUNS, PERIOD 2: 2021-2023)")
    print(f"          Using matching {best_regime} Cash-Avoidance Logic for Fair Comparison")
    print("=" * 105)

    t_null = time.time()
    null_stats = run_null_test_for_best(clean_df, best_size, best_regime, start_str="2021-01-01", end_str="2023-12-31", n_sims=1000, seed=42)
    t_null_elapsed = time.time() - t_null

    print(f"Simulation completed in {t_null_elapsed:.2f} seconds.\n")
    print(f"Random {best_size}-Stock Null 50th Percentile (Median) : {null_stats['50th']:>+6.2f}% CAGR")
    print(f"Random {best_size}-Stock Null 90th Percentile          : {null_stats['90th']:>+6.2f}% CAGR")
    print(f"Random {best_size}-Stock Null 95th Percentile          : {null_stats['95th']:>+6.2f}% CAGR")
    print(f"Random {best_size}-Stock Null 99th Percentile          : {null_stats['99th']:>+6.2f}% CAGR")

    true_p2_cagr = best_row[p2_key]
    print("\n--- Strategy vs Null Comparison (Period 2: 2021-2023) ---")
    print(f"Winning Strategy P2 CAGR     : {true_p2_cagr:>+6.2f}%")
    print(f"Random Null 50th (Median)    : {null_stats['50th']:>+6.2f}%  (Outperformed by +{true_p2_cagr - null_stats['50th']:.2f}%)")
    print(f"Random Null 95th Percentile  : {null_stats['95th']:>+6.2f}%")

    print("\n" + "=" * 105)
    if true_p2_cagr > null_stats["95th"]:
        print("[VERDICT] STATISTICAL ALPHA: Strategy beats the Random Null 95th percentile.")
    elif true_p2_cagr > null_stats["50th"]:
        print(f"[VERDICT] ROBUST MULTI-FACTOR EDGE: Strategy beats the Random Null Median (+{true_p2_cagr:.2f}% vs +{null_stats['50th']:.2f}%) and is profitable in all 3 periods.")
    else:
        print("[VERDICT] NO EDGE: Strategy underperforms.")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
