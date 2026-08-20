import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

client = StockHistoricalDataClient(
    API_KEY,
    SECRET_KEY
)


def get_daily_data(symbol, start_date, end_date=None):

    if end_date is None:
        end_date = datetime.now(timezone.utc)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        feed=DataFeed.IEX
    )

    bars = client.get_stock_bars(request)

    df = bars.df.reset_index()

    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df