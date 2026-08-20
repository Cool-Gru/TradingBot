import os
from datetime import datetime, timedelta, timezone
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

client = StockHistoricalDataClient(
    API_KEY,
    SECRET_KEY
)

symbol = "AAPL"

end = datetime.now(timezone.utc)
start = end - timedelta(days=120)

request = StockBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=TimeFrame.Day,
    start=start,
    end=end,
    feed=DataFeed.IEX
)

bars = client.get_stock_bars(request)
df = bars.df.reset_index()

df["SMA_20"] = df["close"].rolling(window=20).mean()
df["SMA_50"] = df["close"].rolling(window=50).mean()

latest = df.iloc[-1]

print("\n--- CURRENT SIGNAL ---")
print("Symbol:", symbol)
print("Price:", latest["close"])
print("SMA 20:", latest["SMA_20"])
print("SMA 50:", latest["SMA_50"])

if latest["SMA_20"] > latest["SMA_50"]:
    signal = "BUY"

elif latest["SMA_20"] < latest["SMA_50"]:
    signal = "SELL"

else:
    signal = "HOLD"

print("Signal:", signal)

for bar in bars[symbol]:
    print(
        bar.timestamp,
        "Open:", bar.open,
        "High:", bar.high,
        "Low:", bar.low,
        "Close:", bar.close,
        "Volume:", bar.volume
    )