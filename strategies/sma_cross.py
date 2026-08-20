def add_signals(
    df,
    fast=20,
    slow=50,
    trend_filter=200
):

    df = df.copy()

    df["SMA_FAST"] = (
        df["close"]
        .rolling(fast)
        .mean()
    )

    df["SMA_SLOW"] = (
        df["close"]
        .rolling(slow)
        .mean()
    )

    df["SMA_TREND"] = (
        df["close"]
        .rolling(trend_filter)
        .mean()
    )

    df["BUY_SIGNAL"] = False
    df["SELL_SIGNAL"] = False

    for i in range(1, len(df)):

        yesterday = df.iloc[i - 1]
        today = df.iloc[i]

        crossed_up = (
            yesterday["SMA_FAST"]
            <= yesterday["SMA_SLOW"]
            and
            today["SMA_FAST"]
            > today["SMA_SLOW"]
        )

        crossed_down = (
            yesterday["SMA_FAST"]
            >= yesterday["SMA_SLOW"]
            and
            today["SMA_FAST"]
            < today["SMA_SLOW"]
        )

        trend_ok = (
            today["close"]
            > today["SMA_TREND"]
        )

        if crossed_up and trend_ok:
            df.loc[i, "BUY_SIGNAL"] = True

        if crossed_down:
            df.loc[i, "SELL_SIGNAL"] = True

    return df