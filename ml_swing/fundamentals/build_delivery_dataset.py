import os
import io
import time
import urllib.parse
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import nselib
from nselib import capital_market
from nselib.libutil import default_header, header

FUNDAMENTALS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_STOCKS_DIR = os.path.join(FUNDAMENTALS_DIR, "raw_stocks")
DELIVERY_RAW_PARQUET = os.path.join(FUNDAMENTALS_DIR, "delivery_raw.parquet")
DELIVERY_DATASET_PARQUET = os.path.join(FUNDAMENTALS_DIR, "delivery_dataset.parquet")

# 77-stock universe from Phase 1 ML swing system
UNIVERSE = [
    'ABB', 'ADANIENT', 'ADANIPORTS', 'ADANIPOWER', 'AMBUJACEM',
    'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJAJFINSV',
    'BAJAJHLDNG', 'BAJFINANCE', 'BANKBARODA', 'BEL', 'BHARTIARTL',
    'BOSCHLTD', 'BPCL', 'BRITANNIA', 'CANBK', 'CGPOWER',
    'CHOLAFIN', 'CIPLA', 'COALINDIA', 'CUMMINSIND', 'DIVISLAB',
    'DLF', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GODREJCP',
    'GRASIM', 'HCLTECH', 'HDFCBANK', 'HINDALCO', 'HINDUNILVR',
    'HINDZINC', 'ICICIBANK', 'IDEA', 'INFY', 'IOC',
    'ITC', 'JINDALSTEL', 'JSWSTEEL', 'KOTAKBANK', 'LT',
    'M&M', 'MARUTI', 'MOTHERSON', 'MUTHOOTFIN', 'NESTLEIND',
    'NTPC', 'ONGC', 'PFC', 'PIDILITIND', 'PNB',
    'POWERGRID', 'RELIANCE', 'SBIN', 'SHRIRAMFIN', 'SIEMENS',
    'SOLARINDS', 'SUNPHARMA', 'TATACONSUM', 'TATAPOWER', 'TATASTEEL',
    'TCS', 'TECHM', 'TITAN', 'TMPV', 'TORNTPHARM',
    'TRENT', 'TVSMOTOR', 'ULTRACEMCO', 'UNIONBANK', 'VEDL',
    'WIPRO', 'ZYDUSLIFE'
]


def bhav_copy_with_delivery_wrapper(trade_date: str) -> pd.DataFrame:
    """
    Wrapper around nselib.capital_market.bhav_copy_with_delivery.
    Accepts trade_date in 'YYYY-MM-DD' or 'DD-MM-YYYY' format.
    """
    try:
        if len(trade_date) == 10 and trade_date[4] == '-':
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d").strftime("%d-%m-%Y")
        return capital_market.bhav_copy_with_delivery(trade_date=trade_date)
    except Exception as e:
        return pd.DataFrame()


def fetch_stock_delivery_data(symbol: str, start_year: int = 2015, end_year: int = 2024, max_retries: int = 3) -> pd.DataFrame:
    """
    Download official NSE price-volume-delivery data for a stock from start_year to end_year.
    Reuses session cookies to maximize throughput and reliability.
    """
    os.makedirs(RAW_STOCKS_DIR, exist_ok=True)
    cache_file = os.path.join(RAW_STOCKS_DIR, f"{symbol}.parquet")
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_parquet(cache_file)
            if not cached_df.empty:
                print(f"[Cache] Loaded {symbol:12s} from cache ({len(cached_df)} rows)", flush=True)
                return cached_df
        except Exception:
            pass

    t0 = time.time()
    query_sym = "TATAMOTORS" if symbol == "TMPV" else symbol
    encoded_sym = urllib.parse.quote(query_sym)

    session = requests.Session()
    # Initialize cookie session from NSE
    cookie_initialized = False
    for attempt in range(4):
        try:
            session.get("https://nsewebsite-staging.nseindia.com/report-detail/eq_security", headers=default_header, timeout=15)
            cookie_initialized = True
            break
        except Exception:
            time.sleep(1.0 + attempt * 1.5)

    all_dfs = []
    for yr in range(start_year, end_year + 1):
        url = f"https://www.nseindia.com/api/historicalOR/generateSecurityWiseHistoricalData?from=01-01-{yr}&to=31-12-{yr}&symbol={encoded_sym}&type=priceVolumeDeliverable&series=ALL&csv=true"
        success = False
        for attempt in range(max_retries):
            try:
                res = session.get(url, headers=header, timeout=20)
                if res.status_code == 200 and len(res.text) > 100:
                    txt = res.text.replace("\x82", "").replace("\ufffd", "In Rs")
                    df = pd.read_csv(io.StringIO(txt))
                    df.columns = [c.replace(" ", "") for c in df.columns]
                    all_dfs.append(df)
                    success = True
                    break
                else:
                    time.sleep(1.0)
                    session.get("https://nsewebsite-staging.nseindia.com/report-detail/eq_security", headers=default_header, timeout=15)
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if not success:
            print(f"[{symbol}] Warning: Failed to fetch data for year {yr}", flush=True)

    if not all_dfs:
        raise RuntimeError(f"Failed to fetch any data for {symbol}")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df["Symbol"] = symbol

    # Clean numeric fields to prevent mixed-type parquet errors
    numeric_cols = [
        "TotalTradedQuantity", "TurnoverInRs", "No.ofTrades", "DeliverableQty",
        "%DlyQttoTradedQty", "PrevClose", "OpenPrice", "HighPrice", "LowPrice",
        "LastPrice", "ClosePrice", "AveragePrice"
    ]
    for col in numeric_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(
                full_df[col].astype(str).str.replace(",", "").replace("-", np.nan),
                errors="coerce"
            )

    full_df.to_parquet(cache_file, index=False)
    elapsed = time.time() - t0
    print(f"[Download] {symbol:12s} completed in {elapsed:5.1f}s ({len(full_df)} rows)", flush=True)
    return full_df


def download_all_stocks(universe, max_workers: int = 4):
    """Download delivery data for all stocks in the universe concurrently."""
    print(f"[Download] Starting download for {len(universe)} stocks (2015-2024) with {max_workers} workers...", flush=True)
    results = {}
    completed_count = 0
    total = len(universe)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks with slight stagger to avoid concurrent burst on session init
        futures = {}
        for idx, sym in enumerate(universe):
            f = executor.submit(fetch_stock_delivery_data, sym, 2015, 2024)
            futures[f] = sym
            if idx < max_workers:
                time.sleep(0.5)

        for future in as_completed(futures):
            sym = futures[future]
            completed_count += 1
            try:
                df = future.result()
                results[sym] = df
                print(f"[Progress] ({completed_count}/{total}) stocks processed.", flush=True)
            except Exception as e:
                print(f"[Download] {sym:12s} FAILED: {e}", flush=True)

    return results


def process_raw_dataset(stock_data_dict):
    """
    Filter to EQ series, clean fields, compute delivery_qty and delivery_pct,
    and save raw parquet.
    """
    print("[Processing] Consolidating and cleaning raw delivery records...", flush=True)
    cleaned_dfs = []
    for sym, df in stock_data_dict.items():
        if df.empty:
            continue
        df_copy = df.copy()
        
        # Clean Series column
        if "Series" in df_copy.columns:
            df_copy["Series"] = df_copy["Series"].astype(str).str.strip()
            df_copy = df_copy[df_copy["Series"] == "EQ"].copy()

        # Date format in NSE historical data is '%d-%b-%Y' (e.g., '01-Jan-2015')
        df_copy["trade_date"] = pd.to_datetime(df_copy["Date"], format="%d-%b-%Y")
        df_copy["symbol"] = sym

        # Parse numeric columns
        df_copy["close"] = pd.to_numeric(df_copy["ClosePrice"].astype(str).str.replace(",", ""), errors="coerce")
        df_copy["open"] = pd.to_numeric(df_copy["OpenPrice"].astype(str).str.replace(",", ""), errors="coerce")
        df_copy["high"] = pd.to_numeric(df_copy["HighPrice"].astype(str).str.replace(",", ""), errors="coerce")
        df_copy["low"] = pd.to_numeric(df_copy["LowPrice"].astype(str).str.replace(",", ""), errors="coerce")
        df_copy["total_traded_qty"] = pd.to_numeric(df_copy["TotalTradedQuantity"].astype(str).str.replace(",", ""), errors="coerce")
        df_copy["delivery_qty"] = pd.to_numeric(df_copy["DeliverableQty"].astype(str).str.replace(",", ""), errors="coerce")

        # Calculate delivery_pct = (Delivery Qty / Total Traded Qty) * 100
        df_copy["delivery_pct"] = np.where(
            df_copy["total_traded_qty"] > 0,
            (df_copy["delivery_qty"] / df_copy["total_traded_qty"]) * 100.0,
            np.nan
        )

        cols_to_keep = ["trade_date", "symbol", "close", "open", "high", "low", "total_traded_qty", "delivery_qty", "delivery_pct"]
        cleaned_dfs.append(df_copy[cols_to_keep])

    combined_raw = pd.concat(cleaned_dfs, ignore_index=True)

    # Filter to date range 2015-01-01 to 2024-12-31
    combined_raw = combined_raw[
        (combined_raw["trade_date"] >= "2015-01-01") &
        (combined_raw["trade_date"] <= "2024-12-31")
    ]

    # Deduplicate: Ensure no duplicate dates for the same stock
    combined_raw = combined_raw.drop_duplicates(subset=["symbol", "trade_date"]).sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # Save delivery_raw.parquet
    combined_raw.to_parquet(DELIVERY_RAW_PARQUET, index=False)
    print(f"[Raw] Saved raw data to {DELIVERY_RAW_PARQUET} (Shape: {combined_raw.shape})", flush=True)
    return combined_raw


def compute_features_and_target(raw_df):
    """
    Calculate Features (No Lookahead):
    - 20-day rolling average of delivery_pct
    - Z-score of today's delivery % vs its 20-day average
    Target Variable:
    - 5-day forward return: (Close[t+5] / Close[t] - 1)
    """
    print("[Features] Computing rolling delivery features and 5-day forward return target...", flush=True)
    df = raw_df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # 1. 20-day rolling average of delivery_pct (no lookahead, computed per stock)
    df["delivery_pct_20d_avg"] = df.groupby("symbol")["delivery_pct"].transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )

    # 2. 20-day rolling std of delivery_pct
    df["delivery_pct_20d_std"] = df.groupby("symbol")["delivery_pct"].transform(
        lambda s: s.rolling(window=20, min_periods=20).std()
    )

    # 3. Z-score of today's delivery % vs its 20-day average
    # Z = (delivery_pct - delivery_pct_20d_avg) / delivery_pct_20d_std
    df["delivery_pct_zscore"] = np.where(
        df["delivery_pct_20d_std"] > 0,
        (df["delivery_pct"] - df["delivery_pct_20d_avg"]) / df["delivery_pct_20d_std"],
        np.nan
    )

    # 4. Target: 5-day forward return = Close[t+5] / Close[t] - 1
    df["fwd_5d_return"] = df.groupby("symbol")["close"].transform(
        lambda s: s.shift(-5) / s - 1.0
    )
    df["target"] = df["fwd_5d_return"]

    # Final sort and reset
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # Save delivery_dataset.parquet
    df.to_parquet(DELIVERY_DATASET_PARQUET, index=False)
    print(f"[Dataset] Saved final dataset to {DELIVERY_DATASET_PARQUET} (Shape: {df.shape})", flush=True)
    return df


def run_checks_and_testing(df):
    """Perform crucial validation checks and testing on the final dataset."""
    print("\n" + "=" * 60, flush=True)
    print("                    DATASET VALIDATION                    ", flush=True)
    print("=" * 60, flush=True)
    print(f"Dataset Shape: {df.shape}", flush=True)
    print(f"Number of Stocks: {df['symbol'].nunique()}", flush=True)
    print(f"Date Range: {df['trade_date'].min().strftime('%Y-%m-%d')} to {df['trade_date'].max().strftime('%Y-%m-%d')}", flush=True)

    # Crucial Check 1: Datetime object check
    is_datetime = pd.api.types.is_datetime64_any_dtype(df["trade_date"])
    print(f"Check 1 - trade_date is datetime object: {'PASSED' if is_datetime else 'FAILED'} (dtype={df['trade_date'].dtype})", flush=True)
    assert is_datetime, "trade_date column is not parsed as a datetime object!"

    # Crucial Check 2: No duplicate dates for the same stock
    dup_count = df.duplicated(subset=["symbol", "trade_date"]).sum()
    print(f"Check 2 - No duplicate dates per stock: {'PASSED' if dup_count == 0 else 'FAILED'} (duplicates={dup_count})", flush=True)
    assert dup_count == 0, f"Found {dup_count} duplicate (symbol, trade_date) pairs!"

    # Universe Check
    missing_symbols = set(UNIVERSE) - set(df["symbol"].unique())
    print(f"Check 3 - 77 Stock Universe completeness: {'PASSED' if len(missing_symbols) == 0 else 'WARNING'}", flush=True)
    if missing_symbols:
        print(f"  Missing symbols ({len(missing_symbols)}): {sorted(missing_symbols)}", flush=True)

    print("\nFirst 5 rows of final dataset:", flush=True)
    print("-" * 60, flush=True)
    print(df.head(5).to_string(), flush=True)
    print("-" * 60, flush=True)


if __name__ == "__main__":
    stock_data = download_all_stocks(UNIVERSE, max_workers=4)
    raw_df = process_raw_dataset(stock_data)
    final_df = compute_features_and_target(raw_df)
    run_checks_and_testing(final_df)
