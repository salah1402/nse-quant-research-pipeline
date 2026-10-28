"""
Master Backtest Runner.
Executes the winning Multi-Factor Backtest (Momentum + Delivery + 50 SMA Regime Filter).
Usage:
    python backtest.py
    python backtest.py --mode optimize
    python backtest.py --mode monthly
    python backtest.py --mode 5day
"""
import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Quantitative Research Master Backtest Runner")
    parser.add_argument(
        "--mode",
        choices=["multifactor", "optimize", "monthly", "5day"],
        default="multifactor",
        help="Backtest mode to run (default: multifactor)"
    )
    args = parser.parse_args()

    if args.mode == "multifactor":
        script = os.path.join(BASE_DIR, "ml_swing", "fundamentals", "multifactor_backtest.py")
    elif args.mode == "optimize":
        script = os.path.join(BASE_DIR, "ml_swing", "fundamentals", "optimize_multifactor.py")
    elif args.mode == "monthly":
        script = os.path.join(BASE_DIR, "ml_swing", "fundamentals", "backtest_delivery_monthly.py")
    elif args.mode == "5day":
        script = os.path.join(BASE_DIR, "ml_swing", "fundamentals", "backtest_delivery.py")

    print(f"[Runner] Executing {os.path.relpath(script, BASE_DIR)}...\n")
    with open(script, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, {"__name__": "__main__", "__file__": script})
