import os
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

DATASET_PATH = os.path.join(os.path.dirname(__file__), "delivery_dataset.parquet")


def run_ic_test_and_simulation():
    print("=" * 65)
    print("        INFORMATION COEFFICIENT (IC) TEST & NULL SIMULATION       ")
    print("=" * 65)
    
    # -------------------------------------------------------------
    # Task 1: Information Coefficient (IC) Test
    # -------------------------------------------------------------
    print(f"\n[Task 1] Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run build_delivery_dataset.py first.")
        
    df = pd.read_parquet(DATASET_PATH)
    initial_rows = len(df)
    
    # Drop rows where delivery_pct_zscore or fwd_5d_return is NaN
    df_clean = df.dropna(subset=["delivery_pct_zscore", "fwd_5d_return"]).copy()
    dropped_rows = initial_rows - len(df_clean)
    print(f"[Task 1] Initial rows: {initial_rows:,} | Dropped NaN rows: {dropped_rows:,} | Clean rows: {len(df_clean):,}")
    
    # Compute Spearman rank correlation (IC) for every single trading day
    grouped = df_clean.groupby("trade_date")
    daily_records = []
    
    for trade_date, group in grouped:
        if len(group) >= 5:  # Need at least a few stocks to compute rank correlation
            corr, _ = spearmanr(group["delivery_pct_zscore"], group["fwd_5d_return"])
            if not np.isnan(corr):
                rx = rankdata(group["delivery_pct_zscore"].values)
                ry = rankdata(group["fwd_5d_return"].values)
                # Standardized ranks for high-speed simulation
                std_rx = (rx - np.mean(rx)) / (np.std(rx) + 1e-12)
                std_ry = (ry - np.mean(ry)) / (np.std(ry) + 1e-12)
                daily_records.append({
                    "date": trade_date,
                    "ic": corr,
                    "n_stocks": len(group),
                    "std_rx": std_rx,
                    "std_ry": std_ry
                })
                
    n_days = len(daily_records)
    true_ics = np.array([r["ic"] for r in daily_records])
    
    mean_ic = np.mean(true_ics)
    std_ic = np.std(true_ics, ddof=1)
    se_ic = std_ic / np.sqrt(n_days)
    t_stat = mean_ic / se_ic
    
    print("\n--- Task 1: True Information Coefficient Results ---")
    print(f"Total Trading Days Analyzed (N_days) : {n_days:,}")
    print(f"Mean IC                              : {mean_ic:+.5f}")
    print(f"Standard Deviation of IC (std_IC)    : {std_ic:.5f}")
    print(f"Standard Error of IC (std/sqrt(N))   : {se_ic:.5f}")
    print(f"t-statistic                          : {t_stat:+.4f}")
    
    # -------------------------------------------------------------
    # Task 2: Random Null Simulation (1,000 iterations)
    # -------------------------------------------------------------
    n_simulations = 1000
    print(f"\n[Task 2] Running {n_simulations:,} Random Null Simulations...")
    t0 = time.time()
    
    rng = np.random.default_rng(seed=42)
    random_mean_ics = np.zeros(n_simulations)
    
    # Pre-extract standardized arrays for performance
    day_data = [(r["n_stocks"], r["std_rx"], r["std_ry"]) for r in daily_records]
    
    for sim_idx in range(n_simulations):
        sim_ics = np.empty(n_days)
        for d_idx, (n_s, sx, sy) in enumerate(day_data):
            # Randomly shuffle delivery_pct_zscore ranks among stocks for that day
            shuffled_sx = rng.permutation(sx)
            sim_ics[d_idx] = np.dot(shuffled_sx, sy) / (n_s - 1)
        random_mean_ics[sim_idx] = np.mean(sim_ics)
        
    sim_time = time.time() - t0
    
    pct_50 = np.percentile(random_mean_ics, 50)
    pct_90 = np.percentile(random_mean_ics, 90)
    pct_95 = np.percentile(random_mean_ics, 95)
    pct_99 = np.percentile(random_mean_ics, 99)
    
    print(f"[Task 2] 1,000 simulations completed in {sim_time:.2f}s.")
    print("\n--- Task 2: Random Null Simulation Results ---")
    print(f"Random Mean IC 50th Percentile (Median) : {pct_50:+.5f}")
    print(f"Random Mean IC 90th Percentile          : {pct_90:+.5f}")
    print(f"Random Mean IC 95th Percentile          : {pct_95:+.5f}")
    print(f"Random Mean IC 99th Percentile          : {pct_99:+.5f}")
    print(f"Max Random Mean IC in 1,000 Sims        : {np.max(random_mean_ics):+.5f}")
    
    # -------------------------------------------------------------
    # Task 3: Verdict
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("                           VERDICT                         ")
    print("=" * 65)
    print(f"Comparison: True Mean IC ({mean_ic:+.5f}) vs Random 95th Percentile ({pct_95:+.5f})")
    print(f"Significance: t-statistic ({t_stat:+.4f}) vs Threshold (2.0000)")
    
    signal_detected = (mean_ic > pct_95) and (t_stat > 2.0)
    
    if signal_detected:
        print("\n[VERDICT] SIGNAL DETECTED. PROCEED TO MODELING.")
    else:
        print("\n[VERDICT] NO EDGE. THE DELIVERY HYPOTHESIS IS DEAD. DO NOT TRAIN A MODEL.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_ic_test_and_simulation()
