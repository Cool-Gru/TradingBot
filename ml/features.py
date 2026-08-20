import numpy as np


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


def build_features(df):
    df = df.copy()

    # ==========================================
    # RETURNS
    # ==========================================

    df["return_1d"] = (
        df["close"].pct_change(1)
    )

    df["return_5d"] = (
        df["close"].pct_change(5)
    )

    df["return_10d"] = (
        df["close"].pct_change(10)
    )

    df["return_20d"] = (
        df["close"].pct_change(20)
    )


    # ==========================================
    # MOVING AVERAGES
    # ==========================================

    df["sma_10"] = (
        df["close"]
        .rolling(10)
        .mean()
    )

    df["sma_20"] = (
        df["close"]
        .rolling(20)
        .mean()
    )

    df["sma_50"] = (
        df["close"]
        .rolling(50)
        .mean()
    )

    df["sma_100"] = (
        df["close"]
        .rolling(100)
        .mean()
    )


    # ==========================================
    # PRICE VS MOVING AVERAGES
    # ==========================================

    df["price_vs_sma10"] = (
        df["close"] /
        df["sma_10"] - 1
    )

    df["price_vs_sma20"] = (
        df["close"] /
        df["sma_20"] - 1
    )

    df["price_vs_sma50"] = (
        df["close"] /
        df["sma_50"] - 1
    )

    df["price_vs_sma100"] = (
        df["close"] /
        df["sma_100"] - 1
    )


    # ==========================================
    # VOLATILITY
    # ==========================================

    df["volatility_10"] = (
        df["return_1d"]
        .rolling(10)
        .std()
    )

    df["volatility_20"] = (
        df["return_1d"]
        .rolling(20)
        .std()
    )


    # ==========================================
    # RSI
    # ==========================================

    df["rsi_14"] = calculate_rsi(
        df["close"],
        14
    )


    # ==========================================
    # VOLUME
    # ==========================================

    df["volume_avg_20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg_20"]
    )


    # ==========================================
    # DAILY RANGE
    # ==========================================

    df["daily_range"] = (
        df["high"] - df["low"]
    ) / df["close"]


    # ==========================================
    # FUTURE RETURN
    # ==========================================

    # Future 5-trading-day return
    df["future_return_5d"] = (
        df["close"]
        .shift(-5) /
        df["close"] - 1
    )


    # ==========================================
    # TARGET
    # ==========================================

    # 1 = stock gains MORE than 1%
    # over the next 5 trading days
    #
    # 0 = stock gains <= 1% or falls

    df["target"] = (
        df["future_return_5d"] > 0.01
    ).astype(int)


    # ==========================================
    # CLEAN DATA
    # ==========================================

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.dropna(
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df