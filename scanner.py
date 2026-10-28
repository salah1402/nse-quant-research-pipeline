"""
Phase 1 Intraday Scanner Runner.
Provides access to the intraday ORB / Breakout scanner simulation and alerting logic.
"""
import os
import sys

if __name__ == "__main__":
    mock_scanner_path = os.path.join(os.path.dirname(__file__), "mock_scanner.py")
    if os.path.exists(mock_scanner_path):
        with open(mock_scanner_path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code)
    else:
        print(f"Error: {mock_scanner_path} not found.")
