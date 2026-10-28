from openchart import NSEData

nse = NSEData()
nse.download()  # only needed first time

df = nse.historical("RELIANCE-EQ", interval="5m", lookback=5)
print("Total rows:", len(df))
print(df.tail(3))
print("Last timestamp:", df.index[-1] if hasattr(df, "index") else df.iloc[-1, 0])
print("Last volume:", df["Volume"].iloc[-1] if "Volume" in df.columns else "NO VOLUME COLUMN")
