import os
import io
import time
import urllib.parse
from datetime import datetime
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NIFTY100_LIST_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
UPSTOX_NSE_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"

DATE_CHUNKS = [
    ("2014-01-01", "2019-06-30"),
    ("2019-07-01", "2024-12-31")
]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def fetch_nifty100_constituents():
    """Fetch the Nifty 100 constituent symbols from NSE archives or fallback."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    cache_path = os.path.join(DATA_DIR, "nifty100_constituents.csv")
    
    if os.path.exists(cache_path):
        print(f"[Universe] Loading cached Nifty 100 list from {cache_path}")
        return pd.read_csv(cache_path)
    
    print("[Universe] Fetching Nifty 100 constituents from NSE...")
    try:
        resp = requests.get(NIFTY100_LIST_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        ensure_dir(DATA_DIR)
        df.to_csv(cache_path, index=False)
        print(f"[Universe] Successfully fetched {len(df)} constituents.")
        return df
    except Exception as e:
        print(f"[Universe] Failed to fetch from NSE ({e}). Checking local cache...")
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)
        raise

def fetch_upstox_instrument_map():
    """Download Upstox NSE instruments to map trading symbols to instrument keys."""
    cache_path = os.path.join(DATA_DIR, "upstox_nse_instruments.csv")
    if os.path.exists(cache_path):
        print(f"[Instruments] Loading cached Upstox instruments from {cache_path}")
        return pd.read_csv(cache_path)
    
    print("[Instruments] Downloading Upstox NSE instrument master file...")
    ensure_dir(DATA_DIR)
    nse_df = pd.read_csv(UPSTOX_NSE_INSTRUMENTS_URL)
    nse_df.to_csv(cache_path, index=False)
    print(f"[Instruments] Downloaded {len(nse_df)} instruments.")
    return nse_df

def fetch_candles_from_upstox(instrument_key, date_chunks=DATE_CHUNKS, max_retries=3):
    """
    Fetch historical daily candles for an instrument key across date chunks.
    Upstox candle format: [timestamp, open, high, low, close, volume, open_interest]
    """
    all_rows = []
    encoded_key = urllib.parse.quote(instrument_key)
    
    for from_d, to_d in date_chunks:
        url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/day/{to_d}/{from_d}"
        success = False
        
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    candles = data.get("data", {}).get("candles", [])
                    all_rows.extend(candles)
                    success = True
                    break
                elif resp.status_code == 429:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    time.sleep(0.5)
            except Exception as e:
                time.sleep(0.5)
                
        if not success:
            print(f"  Warning: failed to fetch chunk {from_d} to {to_d} for {instrument_key}")
            
    if not all_rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]

def download_all_data():
    ensure_dir(DATA_DIR)
    
    # 1. Fetch NIFTY 50 Index first
    nifty50_cache = os.path.join(DATA_DIR, "NIFTY50_daily.csv")
    if not os.path.exists(nifty50_cache):
        print("[Data] Downloading NIFTY 50 Index data (2014-01-01 to 2024-12-31)...")
        nifty_df = fetch_candles_from_upstox("NSE_INDEX|Nifty 50")
        if not nifty_df.empty:
            nifty_df.to_csv(nifty50_cache, index=False)
            print(f"[Data] Cached NIFTY 50: {len(nifty_df)} trading days.")
        else:
            print("[Data] ERROR: Failed to download NIFTY 50 index data!")
    else:
        nifty_df = pd.read_csv(nifty50_cache)
        print(f"[Data] Found cached NIFTY 50 ({len(nifty_df)} rows).")

    # 2. Fetch Universe & Instrument mapping
    n100_df = fetch_nifty100_constituents()
    inst_df = fetch_upstox_instrument_map()
    
    eq_inst = inst_df[inst_df["instrument_type"] == "EQUITY"].copy()
    symbol_to_key = dict(zip(eq_inst["tradingsymbol"], eq_inst["instrument_key"]))
    
    symbols = n100_df["Symbol"].unique().tolist()
    print(f"\n[Data] Starting download for {len(symbols)} universe stocks...")
    
    success_count = 0
    skip_count = 0
    
    for i, symbol in enumerate(symbols, 1):
        target_path = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
        if os.path.exists(target_path):
            existing_df = pd.read_csv(target_path)
            if len(existing_df) > 500:
                skip_count += 1
                continue
                
        inst_key = symbol_to_key.get(symbol)
        if not inst_key:
            # Try searching by ISIN
            isin_match = n100_df[n100_df["Symbol"] == symbol]
            if not isin_match.empty:
                isin = isin_match.iloc[0]["ISIN Code"]
                inst_key = f"NSE_EQ|{isin}"
                
        if not inst_key:
            print(f"[{i}/{len(symbols)}] Skipping {symbol}: No Upstox instrument key found.")
            continue
            
        print(f"[{i}/{len(symbols)}] Downloading {symbol} ({inst_key})...")
        df = fetch_candles_from_upstox(inst_key)
        if not df.empty:
            df.to_csv(target_path, index=False)
            success_count += 1
        else:
            print(f"  Warning: No data returned for {symbol}")
            
        # Mild throttle to prevent API rate limiting
        time.sleep(0.1)
        
    print(f"\n[Data Download Complete] Successfully downloaded/updated {success_count} stocks, {skip_count} cached.")

if __name__ == "__main__":
    download_all_data()
