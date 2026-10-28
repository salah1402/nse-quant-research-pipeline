"""
Phase 3 & Phase 4: Baseline Strategy and Walk-Forward ML Model Simulation.
"""

import os
import glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.parquet")
BASELINE_SUMMARY_PATH = os.path.join(BASE_DIR, "baseline_summary.txt")
ML_SUMMARY_PATH = os.path.join(BASE_DIR, "ml_summary.txt")

# Realistic delivery transaction cost per side in Indian markets:
# - Slippage: 0.05% (0.0005)
# - STT on delivery: 0.10% (0.0010)
# - Brokerage & other taxes (Exchange turnover, GST, Stamp duty, SEBI): 0.02% (0.0002)
COST_PER_SIDE = 0.0017  # 0.17% per side, ~0.34% round-trip per position turnover

FEATURES = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m",
    "volatility_20d",
    "volume_pct_change",
    "rs_vs_nifty_3m",
    "dist_from_52w_high"
]

def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run build_features.py first.")
    
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    
    # Load daily closing prices for exact rebalancing simulation
    price_dict = {}
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_daily.csv"))
    for file_path in csv_files:
        sym = os.path.basename(file_path).replace("_daily.csv", "")
        stock_df = pd.read_csv(file_path, usecols=["date", "close"])
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        price_dict[sym] = stock_df.set_index("date")["close"]
        
    prices = pd.DataFrame(price_dict).ffill()
    return df, prices

def calculate_performance_metrics(monthly_rets, trade_rets):
    capital = 1.0
    equity_curve = [capital]
    for r in monthly_rets:
        capital *= (1.0 + r)
        equity_curve.append(capital)
        
    n_months = len(monthly_rets)
    years = n_months / 12.0
    cagr = (capital ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    max_dd = np.min(drawdowns) * 100.0
    
    monthly_win_rate = (np.sum(np.array(monthly_rets) > 0) / len(monthly_rets)) * 100.0 if monthly_rets else 0.0
    trade_win_rate = (np.sum(np.array(trade_rets) > 0) / len(trade_rets)) * 100.0 if trade_rets else 0.0
    
    return {
        "cagr": cagr,
        "max_dd": max_dd,
        "monthly_win_rate": monthly_win_rate,
        "trade_win_rate": trade_win_rate,
        "final_equity": capital,
        "equity_curve": equity_curve
    }

def run_baseline_strategy(df, prices, month_ends, start_idx, end_idx):
    monthly_rets = []
    trade_rets = []
    prev_holdings = set()
    
    for i in range(start_idx, end_idx):
        d_curr = month_ends[i]
        d_next = month_ends[i + 1]
        
        # Rank by 12M momentum at month-end and pick top 10
        day_slice = df[df["date"] == d_curr].sort_values("momentum_12m", ascending=False)
        top10 = day_slice.head(10)["symbol"].tolist()
        curr_holdings = set(top10)
        
        # Calculate monthly holding period return with realistic costs
        period_stock_rets = []
        for sym in top10:
            p_entry = prices.loc[d_curr, sym]
            p_exit = prices.loc[d_next, sym]
            gross_ret = (p_exit / p_entry) - 1.0
            
            # Apply cost: entry if new position, exit cost
            cost = COST_PER_SIDE
            if sym not in prev_holdings:
                cost += COST_PER_SIDE
                
            net_ret = gross_ret - cost
            period_stock_rets.append(net_ret)
            trade_rets.append(net_ret)
            
        monthly_rets.append(np.mean(period_stock_rets))
        prev_holdings = curr_holdings
        
    return calculate_performance_metrics(monthly_rets, trade_rets)

def run_walk_forward_ml(df, prices, month_ends, date_to_idx):
    folds = [
        ("Fold 1", list(range(2015, 2020)), 2020, pd.to_datetime("2019-12-31"), pd.to_datetime("2020-12-31")),
        ("Fold 2", list(range(2016, 2021)), 2021, pd.to_datetime("2020-12-31"), pd.to_datetime("2021-12-31")),
        ("Fold 3", list(range(2017, 2022)), 2022, pd.to_datetime("2021-12-31"), pd.to_datetime("2022-12-30")),
        ("Fold 4", list(range(2018, 2023)), 2023, pd.to_datetime("2022-12-30"), pd.to_datetime("2023-12-29")),
    ]
    
    all_monthly_rets = []
    all_trade_rets = []
    last_trained_model = None
    
    print("\nStarting Walk-Forward ML Model Simulation (2020-2023):")
    print("-" * 65)
    
    for fold_name, train_years, test_year, start_d, end_d in folds:
        # Prevent forward target lookahead: train rows must not use forward prices from test period
        train_mask = df["date"].dt.year.isin(train_years) & (df["date"] < start_d - pd.Timedelta(days=7))
        X_train = df.loc[train_mask, FEATURES]
        y_train = df.loc[train_mask, "target"]
        
        print(f"[{fold_name}] Training XGBoost on {min(train_years)}-{max(train_years)} ({len(X_train)} samples)...")
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        last_trained_model = model
        
        s_idx = date_to_idx[start_d]
        e_idx = date_to_idx[end_d]
        
        prev_holdings = set()
        fold_monthly_rets = []
        
        for i in range(s_idx, e_idx):
            d_curr = month_ends[i]
            d_next = month_ends[i + 1]
            
            day_slice = df[df["date"] == d_curr].copy()
            # Predict probabilities of beating Nifty 50
            day_slice["pred_prob"] = model.predict_proba(day_slice[FEATURES])[:, 1]
            top10 = day_slice.sort_values("pred_prob", ascending=False).head(10)["symbol"].tolist()
            curr_holdings = set(top10)
            
            period_stock_rets = []
            for sym in top10:
                p_entry = prices.loc[d_curr, sym]
                p_exit = prices.loc[d_next, sym]
                gross_ret = (p_exit / p_entry) - 1.0
                
                cost = COST_PER_SIDE
                if sym not in prev_holdings:
                    cost += COST_PER_SIDE
                    
                net_ret = gross_ret - cost
                period_stock_rets.append(net_ret)
                all_trade_rets.append(net_ret)
                
            m_ret = np.mean(period_stock_rets)
            fold_monthly_rets.append(m_ret)
            all_monthly_rets.append(m_ret)
            prev_holdings = curr_holdings
            
        fold_cagr = (np.prod([1.0 + r for r in fold_monthly_rets]) - 1.0) * 100.0
        print(f"  -> Test Year {test_year}: 12 months simulated. 1-Year Return: {fold_cagr:.2f}%")
        
    metrics = calculate_performance_metrics(all_monthly_rets, all_trade_rets)
    return metrics, last_trained_model

def main():
    print("=" * 65)
    print("PHASE 3 & 4: STRATEGY BACKTEST & ML WALK-FORWARD SIMULATION")
    print("=" * 65)
    
    df, prices = load_data()
    
    # Identify month-end trading days
    month_ends = df.groupby([df["date"].dt.year, df["date"].dt.month])["date"].max().reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(month_ends)}
    
    # Define test range (2020-2023: 2019-12-31 to 2023-12-29, 48 monthly periods)
    start_test_date = pd.to_datetime("2019-12-31")
    end_test_date = pd.to_datetime("2023-12-29")
    start_idx = date_to_idx[start_test_date]
    end_idx = date_to_idx[end_test_date]
    
    # -------------------------------------------------------------
    # PHASE 3: THE BASELINE (12M Momentum Dumb Rule)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 3: BASELINE MOMENTUM STRATEGY (12M MOMENTUM, TOP 10)")
    print("=" * 65)
    
    # 1. Baseline on Test Period (2020-2023) for direct comparison
    baseline_test = run_baseline_strategy(df, prices, month_ends, start_idx, end_idx)
    
    # 2. Baseline on Full Period (2015-2023)
    baseline_full = run_baseline_strategy(df, prices, month_ends, 0, end_idx)
    
    print("\nBaseline Results on Walk-Forward Test Period (2020-2023):")
    print(f"  CAGR:             {baseline_test['cagr']:.2f}%")
    print(f"  Max Drawdown:     {baseline_test['max_dd']:.2f}%")
    print(f"  Monthly Win Rate: {baseline_test['monthly_win_rate']:.2f}%")
    print(f"  Trade Win Rate:   {baseline_test['trade_win_rate']:.2f}%")
    print(f"  Final Equity:     {baseline_test['final_equity']:.4f}x")
    
    print("\nBaseline Results on Full Historical Period (2015-2023):")
    print(f"  CAGR:             {baseline_full['cagr']:.2f}%")
    print(f"  Max Drawdown:     {baseline_full['max_dd']:.2f}%")
    print(f"  Monthly Win Rate: {baseline_full['monthly_win_rate']:.2f}%")
    print(f"  Trade Win Rate:   {baseline_full['trade_win_rate']:.2f}%")
    print(f"  Final Equity:     {baseline_full['final_equity']:.4f}x")
    
    # Save baseline summary to file
    with open(BASELINE_SUMMARY_PATH, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PHASE 3: BASELINE STRATEGY SUMMARY\n")
        f.write("Rule: Monthly Rebalance, Top 10 by 12M Momentum, Equal Weight\n")
        f.write(f"Costs: Delivery cost {COST_PER_SIDE*100:.2f}% per side (STT, slippage 0.05%, fees)\n")
        f.write("=" * 60 + "\n\n")
        f.write("[WALK-FORWARD TEST PERIOD: 2020 - 2023 (48 Months)]\n")
        f.write(f"Baseline CAGR:             {baseline_test['cagr']:.2f}%\n")
        f.write(f"Baseline Max Drawdown:     {baseline_test['max_dd']:.2f}%\n")
        f.write(f"Baseline Monthly Win Rate: {baseline_test['monthly_win_rate']:.2f}%\n")
        f.write(f"Baseline Trade Win Rate:   {baseline_test['trade_win_rate']:.2f}%\n")
        f.write(f"Baseline Final Equity:     {baseline_test['final_equity']:.4f}x\n\n")
        f.write("[FULL HISTORICAL PERIOD: 2015 - 2023 (107 Months)]\n")
        f.write(f"Baseline CAGR:             {baseline_full['cagr']:.2f}%\n")
        f.write(f"Baseline Max Drawdown:     {baseline_full['max_dd']:.2f}%\n")
        f.write(f"Baseline Monthly Win Rate: {baseline_full['monthly_win_rate']:.2f}%\n")
        f.write(f"Baseline Trade Win Rate:   {baseline_full['trade_win_rate']:.2f}%\n")
        f.write(f"Baseline Final Equity:     {baseline_full['final_equity']:.4f}x\n")
        
    print(f"\n[Saved] Baseline summary written to: {BASELINE_SUMMARY_PATH}")
    
    # -------------------------------------------------------------
    # PHASE 4: THE ML MODEL (WALK-FORWARD VALIDATION 2020-2023)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 4: ML MODEL (WALK-FORWARD XGBOOST, 2020-2023)")
    print("=" * 65)
    
    ml_metrics, last_model = run_walk_forward_ml(df, prices, month_ends, date_to_idx)
    
    print("\n" + "=" * 65)
    print("ML MODEL WALK-FORWARD SUMMARY (2020-2023)")
    print("=" * 65)
    print(f"  ML Model CAGR:             {ml_metrics['cagr']:.2f}%")
    print(f"  ML Model Max Drawdown:     {ml_metrics['max_dd']:.2f}%")
    print(f"  ML Model Monthly Win Rate: {ml_metrics['monthly_win_rate']:.2f}%")
    print(f"  ML Model Trade Win Rate:   {ml_metrics['trade_win_rate']:.2f}%")
    print(f"  ML Model Final Equity:     {ml_metrics['final_equity']:.4f}x")
    
    # Feature Importances from the last trained model (Fold 4: 2018-2022)
    fi = pd.Series(last_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature Importance (Last Trained Model: Train 2018-2022):")
    print("-" * 65)
    for feat, imp in fi.items():
        print(f"  {feat:<22}: {imp*100:.2f}%")
        
    # Save ML summary to file
    with open(ML_SUMMARY_PATH, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PHASE 4: ML MODEL WALK-FORWARD VALIDATION SUMMARY\n")
        f.write("Model: XGBClassifier (Walk-Forward Folds: 2020, 2021, 2022, 2023)\n")
        f.write("Strategy: Top 10 by predicted probability, monthly rebalance\n")
        f.write(f"Costs: Delivery cost {COST_PER_SIDE*100:.2f}% per side (STT, slippage 0.05%, fees)\n")
        f.write("Holdout: 2024 completely untouched\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"ML Model CAGR:             {ml_metrics['cagr']:.2f}%\n")
        f.write(f"ML Model Max Drawdown:     {ml_metrics['max_dd']:.2f}%\n")
        f.write(f"ML Model Monthly Win Rate: {ml_metrics['monthly_win_rate']:.2f}%\n")
        f.write(f"ML Model Trade Win Rate:   {ml_metrics['trade_win_rate']:.2f}%\n")
        f.write(f"ML Model Final Equity:     {ml_metrics['final_equity']:.4f}x\n\n")
        f.write("Feature Importance (Last Model: Train 2018-2022):\n")
        f.write("-" * 45 + "\n")
        for feat, imp in fi.items():
            f.write(f"{feat:<22}: {imp*100:.2f}%\n")
            
    print(f"\n[Saved] ML model summary written to: {ML_SUMMARY_PATH}")
    print("=" * 65)

if __name__ == "__main__":
    main()
