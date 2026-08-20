import pandas as pd


def build_market_features(stock_df, spy_df, qqq_df):

    stock_df = stock_df.copy()
    spy_df = spy_df.copy()
    qqq_df = qqq_df.copy()


    # ==========================================
    # SPY MARKET FEATURES
    # ==========================================

    spy_df["spy_return_5d"] = (
        spy_df["close"].pct_change(5)
    )

    spy_df["spy_return_20d"] = (
        spy_df["close"].pct_change(20)
    )

    spy_df["spy_sma50"] = (
        spy_df["close"]
        .rolling(50)
        .mean()
    )

    spy_df["spy_sma200"] = (
        spy_df["close"]
        .rolling(200)
        .mean()
    )

    spy_df["spy_vs_sma50"] = (
        spy_df["close"] /
        spy_df["spy_sma50"] - 1
    )

    spy_df["spy_vs_sma200"] = (
        spy_df["close"] /
        spy_df["spy_sma200"] - 1
    )


    # ==========================================
    # QQQ MARKET FEATURES
    # ==========================================

    qqq_df["qqq_return_5d"] = (
        qqq_df["close"].pct_change(5)
    )

    qqq_df["qqq_return_20d"] = (
        qqq_df["close"].pct_change(20)
    )

    qqq_df["qqq_sma50"] = (
        qqq_df["close"]
        .rolling(50)
        .mean()
    )

    qqq_df["qqq_sma200"] = (
        qqq_df["close"]
        .rolling(200)
        .mean()
    )

    qqq_df["qqq_vs_sma50"] = (
        qqq_df["close"] /
        qqq_df["qqq_sma50"] - 1
    )

    qqq_df["qqq_vs_sma200"] = (
        qqq_df["close"] /
        qqq_df["qqq_sma200"] - 1
    )


    # ==========================================
    # KEEP ONLY THESE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ==========================================

    spy_features = spy_df[
        [
            "timestamp",
            "spy_return_5d",
            "spy_return_20d",
            "spy_vs_sma50",
            "spy_vs_sma200"
        ]
    ].copy()


    qqq_features = qqq_df[
        [
            "timestamp",
            "qqq_return_5d",
            "qqq_return_20d",
            "qqq_vs_sma50",
            "qqq_vs_sma200"
        ]
    ].copy()


    # ==========================================
    # MERGE WITH STOCK
    # ==========================================

    result = pd.merge(
        stock_df,
        spy_features,
        on="timestamp",
        how="left"
    )

    result = pd.merge(
        result,
        qqq_features,
        on="timestamp",
        how="left"
    )


    # ==========================================
    # RELATIVE STRENgTH
    # ==========================================

    result["relative_spy_5d"] = (
        result["return_5d"]
        - result["spy_return_5d"]
    )

    result["relative_spy_20d"] = (
        result["return_20d"]
        - result["spy_return_20d"]
    )

    result["relative_qqq_5d"] = (
        result["return_5d"]
        - result["qqq_return_5d"]
    )

    result["relative_qqq_20d"] = (
        result["return_20d"]
        - result["qqq_return_20d"]
    )


    # ==========================================
    # CLEAN
    # ==========================================

    result = result.dropna().copy()

    result = result.reset_index(
        drop=True
    )

    return result