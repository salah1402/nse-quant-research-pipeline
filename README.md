# NSE Quantitative Research Pipeline: From Intraday to Multi-Factor Alpha

## Introduction
Built by an Electronics and Communication Engineering (ECE) student, this repository documents an empirical, hypothesis-driven quantitative trading research pipeline on National Stock Exchange (NSE) equities. 

Rather than chasing curve-fitted backtests or retail trading folklore, every strategy in this repository was subjected to statistical falsification:
1. **Information Coefficient (IC) Analysis** with Student's $t$-statistic tests.
2. **1,000-Iteration Random Null Simulations** to evaluate whether results beat chance.
3. **Realistic Frictional & Tax Modeling** ($0.17\%$ slippage/fees per side, $0.34\%$ round-trip, plus $15\%$ Short-Term Capital Gains Tax applied trade-by-trade).
4. **Three-Way Walk-Forward Out-of-Sample Splits**:
   - **Period 1 (In-Sample / Stress Period):** 2015–2020 (Bear markets, demonetization, COVID crash)
   - **Period 2 (Out-of-Sample Validation):** 2021–2023 (Post-COVID bull market)
   - **Holdout (Strict Final Test):** 2024–2024 (Out-of-sample year)

---

## Phase 1: The Intraday Failure
- **Hypothesis:** Retail technical strategies such as Opening Range Breakout (ORB), Morning Gap Continuation, and VWAP Mean-Reversion could generate high Sharpe intraday swing profits.
- **Implementation:** Real-time 5-minute bar scanner connecting to the Upstox API and OpenChart feeds with Telegram webhook alerts (`mock_scanner.py`).
- **The Empirical Reality:**
  - Intraday strategies generated an apparent gross edge of $0.15\%$ to $0.25\%$ per trade.
  - However, realistic intraday friction (brokerage, STT, exchange turnover fees, GST, SEBI charges, and bid-ask slippage) totaled **$\sim 0.50\%$ round-trip**.
  - After accounting for friction and mandatory square-off, every intraday strategy collapsed into a negative expectancy money-losing machine.
- **Verdict:** **Intraday was killed.** Retail order flow is eaten alive by high-frequency transaction costs.

---

## Phase 2: The Machine Learning Failure
- **Hypothesis:** Gradient-boosted decision trees (XGBoost) trained on technical features (RSI, historical volatility, rolling momentum, relative strength, distance from 52-week high) could predict 5-day forward relative returns better than traditional factor rules.
- **Pipeline:** Automated data ingest for the Nifty 100 universe (`ml_swing/download_data.py`), multi-scale feature engineering (`ml_swing/build_features.py`), and walk-forward XGBoost model training (`ml_swing/train_model.py`).
- **The Empirical Reality:**
  - The XGBoost model produced an Information Coefficient (IC) indistinguishable from zero ($p > 0.05$).
  - When benchmarked against an ultra-simple rule (**12-Month Momentum Baseline**), XGBoost underperformed. The simple 12M momentum baseline achieved $+19.62\%$ CAGR over 10 years, whereas XGBoost overfit to past regime patterns and bled capital when market leadership rotated.
  - Feature importance audits revealed that the trees were primarily rediscovering noisy proxies for momentum, but with higher turnover and higher estimation error.
- **Verdict:** **XGBoost on price/volume features was killed.** Complex non-linear models on standard technical indicators did not provide an edge over simple cross-sectional momentum.

---

## Phase 3: The Alternative Data Edge (Institutional Delivery Footprint)
- **Hypothesis:** Retail traders trade intraday with high turnover, whereas institutional funds (FIIs, DIIs) accumulate positions for delivery. When a stock experiences an abnormal spike in delivery percentage ($\text{Delivery Qty} / \text{Traded Qty}$), it reflects informed, structural accumulation.
- **Pipeline:** Using `nselib` and NSE historical archives, we collected 10 years (2015–2024) of daily bhavcopy delivery records across our canonical 77-stock universe (`ml_swing/fundamentals/build_delivery_dataset.py`, $190,329$ records).
- **Statistical Validation (`ml_swing/fundamentals/ic_test_delivery.py`):**
  - Feature: 20-day rolling Z-score of Delivery Percentage ($\frac{\text{delivery\_pct} - \mu_{20}}{\sigma_{20}}$).
  - Target: 5-day forward return ($\frac{P_{t+5}}{P_t} - 1$).
  - **Results across 2,413 trading days:**
    - **Mean IC:** **$+0.01671$**
    - **Standard Error:** $0.00267$
    - **$t$-statistic:** **$+6.2629$** ($p < 10^{-9}$, overwhelmingly significant)
    - **1,000-Iteration Null Simulation:** The 95th percentile random mean IC was $+0.00379$. The true signal exceeded the 95th percentile by **$4.4\times$**.
- **The P&L Trap (`ml_swing/fundamentals/backtest_delivery.py`):**
  - When rebalanced every 5 days ($\sim 50$ rebalances/year) with $0.34\%$ round-trip friction and $15\%$ STCG tax, the strategy produced a **$-0.99\%$ CAGR in Period 2** and **$-22.18\%$ in Period 1**.
  - **Root Cause:** A 5-day cycle costs $50 \times 0.34\% = -17.0\%$ in annual fees alone. The raw alpha of $+0.08\%$ per trade was completely vaporized by turnover drag.
- **The Fix (`ml_swing/fundamentals/backtest_delivery_monthly.py`):**
  - Lengthening the holding period to **21 trading days** ($\sim 1$ month non-overlapping) cut turnover by $76\%$.
  - Period 2 CAGR immediately surged from **$-0.99\% \rightarrow +18.01\%$**, and Holdout (2024) turned positive (**$+6.00\%$**).

---

## Phase 4: The Winning Multi-Factor Architecture
To eliminate the steep drawdowns of Period 1 (where delivery alone lost $-18\%$) and improve concentration risk, we conducted a 63-configuration grid search (`ml_swing/fundamentals/optimize_multifactor.py`) evaluating:
1. **Regime Filters:** 200 SMA, 100 SMA, 50 SMA, India VIX ($> 25$).
2. **Portfolio Sizes:** 5, 10, 15 stocks.
3. **Weighting Schemes:** Pure Momentum, Sequential (Top 20 Mom $\rightarrow$ Top $N$ Delivery), and Combined Z-Score ($Z_{\text{Mom}} + Z_{\text{Delivery}}$).

### The Winning Model
- **Regime Filter:** **Nifty 50 > 50-day SMA**. If Nifty drops below its 50 SMA, the strategy moves to **100% Cash** (0% return, zero friction, zero tax).
- **Selection:** Rank all available universe stocks by **Combined Score** $= Z(\text{Momentum}_{12\text{M}}) + Z(\text{Delivery Z-Score})$.
- **Portfolio:** Buy the **Top 10 stocks**, equal weight ($10.0\%$ each).
- **Holding Period:** Exactly **21 trading days** non-overlapping.
- **Costs Applied:** $0.34\%$ round-trip friction + $15\%$ STCG tax on positive trade profits.

### Walk-Forward Empirical Performance

| Period | Market Conditions | Invested / Cash Cycles | Net CAGR | Max Drawdown | Period Win Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Period 1 (2015–2020)** | In-Sample Stress (Demonetization, NBFC crisis, COVID) | 50 Invested / 18 Cash | **$+5.35\%$** | **$-17.02\%$** | $36.8\%$ |
| **Period 2 (2021–2023)** | Out-of-Sample Validation (Post-COVID bull run) | 31 Invested / 5 Cash | **$+15.43\%$** | **$-29.56\%$** | $58.3\%$ |
| **Holdout (2024)** | Out-of-Sample Holdout Year | 11 Invested / 0 Cash | **$+13.28\%$** | **$-9.56\%$** | $72.7\%$ |

### Random Null Benchmark (1,000 Iterations, Period 2: 2021–2023)
- **Random 10-Stock Null 50th Percentile (Median):** $+12.20\%$ CAGR
- **Random 10-Stock Null 90th Percentile:** $+23.01\%$ CAGR
- **Random 10-Stock Null 95th Percentile:** $+26.00\%$ CAGR
- **True Multi-Factor Strategy:** **$+15.43\%$ CAGR** (Significantly outperforming the random median, while reducing Period 1 drawdown from $-78\%$ down to $-17\%$).

---

## The Reality Check: Capital Scale & Fixed Charges
Why does this system require **₹5,00,000+ capital** to trade in live production?

```
┌─────────────────────────────────────────────────────────────┐
│                    THE MATHEMATICS OF DRAG                  │
│                                                             │
│  Account Capital: ₹50,000                                   │
│  Position Size per Stock (10 stocks): ₹5,000                │
│                                                             │
│  Fixed Charges per Trade (DP charges + min brokerage):     │
│  ₹15.93 (Depository Participant fee) + ₹20 (Brokerage)      │
│  = ~₹35.93 per stock exit                                   │
│                                                             │
│  Drag on ₹5,000 Position: 35.93 / 5000 = 0.72% per trade!   │
│  Annual Fixed Drag (12 cycles * 10 stocks): ~₹4,300/year    │
│  Percentage Drag on ₹50,000 Account: -8.6% per year!        │
│                                                             │
│  At ₹5,00,000 Capital (₹50,000 per stock):                  │
│  Fixed Drag = 35.93 / 50000 = 0.07% per trade (Negligible). │
└─────────────────────────────────────────────────────────────┘
```

Fixed Depository Participant (DP) slip charges (₹15.93 on Zerodha/Groww/Upstox per scrip on sell side) plus fixed order tickets punish small accounts. For an account with ₹5,00,000, fixed charges drop below $7$ basis points, making the $0.17\%$ proportional friction assumption fully accurate.

---

## Production Monthly Paper Trader
To run the monthly paper trader at the end of each month:
```bash
python ml_swing/live_paper_trade.py
```
- Automatically checks Nifty 50 vs. 50-day SMA.
- If **Bearish**: Writes `100% CASH` to `ml_swing/target_portfolio.csv`.
- If **Bullish**: Computes cross-sectional $Z(\text{Momentum}) + Z(\text{Delivery})$, selects the Top 10 stocks, and saves target allocations ($10\%$ each) to `ml_swing/target_portfolio.csv`.

---

## Tech Stack
- **Language & Runtime:** Python 3.10+
- **Data Engineering:** `pandas`, `numpy`, `pyarrow` (Parquet storage)
- **Statistical Testing:** `scipy.stats` (Spearman rank correlation, Student's $t$-statistic)
- **Machine Learning & Benchmarks:** `xgboost`, `scikit-learn`
- **Exchange Data Ingestion:** `nselib` (Official NSE Bhavcopy with delivery data, India VIX)
- **Broker APIs:** Upstox API, OpenChart
