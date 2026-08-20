def run_backtest(
    df,
    starting_cash=10000,
    slippage=0.0005
):

    cash = starting_cash
    shares = 0

    entry_price = None
    entry_date = None

    trades = []
    equity_history = []

    for i in range(len(df) - 1):

        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]

        # =========================
        # BUY
        # =========================

        if (
            today["BUY_SIGNAL"]
            and shares == 0
        ):

            execution_price = (
                tomorrow["open"]
                * (1 + slippage)
            )

            shares_to_buy = int(
                cash // execution_price
            )

            if shares_to_buy > 0:

                cost = (
                    shares_to_buy
                    * execution_price
                )

                cash -= cost
                shares = shares_to_buy

                entry_price = execution_price
                entry_date = tomorrow["timestamp"]

        # =========================
        # SELL
        # =========================

        elif (
            today["SELL_SIGNAL"]
            and shares > 0
        ):

            execution_price = (
                tomorrow["open"]
                * (1 - slippage)
            )

            cash += (
                shares
                * execution_price
            )

            return_pct = (
                execution_price
                / entry_price
                - 1
            ) * 100

            trades.append({

                "entry_date": entry_date,

                "exit_date":
                    tomorrow["timestamp"],

                "entry_price":
                    entry_price,

                "exit_price":
                    execution_price,

                "return_pct":
                    return_pct,

                "shares":
                    shares
            })

            shares = 0
            entry_price = None
            entry_date = None

        portfolio_value = (
            cash
            + shares * today["close"]
        )

        equity_history.append(
            portfolio_value
        )

    final_price = df.iloc[-1]["close"]

    final_value = (
        cash
        + shares * final_price
    )

    equity_history.append(
        final_value
    )

    return (
        equity_history,
        trades
    )