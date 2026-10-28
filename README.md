# NSE Quantitative Research Pipeline

## Overview
A rigorous Python-based quantitative research framework for the Indian stock market (NSE). Built as a college project to test if it's possible to find a tradable edge at a retail scale using free data.

## The Journey
1. **Intraday Strategies (Failed):** Tested ORB, Gap Continuation, and Reversal. All died to 0.5% round-trip friction.
2. **Price/Volume ML (Failed):** XGBoost on price momentum failed the Information Coefficient (IC) test. Features had zero predictive power.
3. **Alternative Data (The Pivot):** Extracted NSE Bhavcopy Delivery Percentage (Institutional Footprint).
4. **The Discovery:** Delivery Z-score showed a statistically significant IC (t-stat 6.26).
5. **The Strategy:** Top 10 stocks by Combined Score (Z-Score of 12M Momentum + Z-Score of Delivery), filtered by a 50-day SMA Regime Filter (Nifty).

## Performance (Net of 0.34% Friction & 20% STCG Tax)
*   **In-Sample (2015-2020):** +5.35% CAGR
*   **Out-of-Sample (2021-2023):** +15.43% CAGR
*   **Holdout (2024):** +13.28% CAGR (-9.56% MaxDD)

## The Reality Check
While the strategy shows an edge, fixed brokerage and DP charges on a ₹5,000 account create a 7% drag per trade. This strategy requires a minimum of ₹5,00,000 to be mathematically viable. 

## Tech Stack
Python, Pandas, XGBoost, Optuna, NSE Bhavcopy, Upstox API.
