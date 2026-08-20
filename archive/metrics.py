import pandas as pd


def calculate_metrics(
    equity_history,
    trades,
    starting_cash
):

    equity = pd.Series(equity_history)

    final_value = equity.iloc[-1]

    total_return = (
        final_value / starting_cash - 1
    ) * 100

    running_max = equity.cummax()

    drawdown = (
        equity - running_max
    ) / running_max

    max_drawdown = (
        drawdown.min() * 100
    )

    if len(trades) > 0:

        winners = [
            trade
            for trade in trades
            if trade["return_pct"] > 0
        ]

        win_rate = (
            len(winners)
            / len(trades)
        ) * 100

        avg_trade = sum(
            trade["return_pct"]
            for trade in trades
        ) / len(trades)

        best_trade = max(
            trade["return_pct"]
            for trade in trades
        )

        worst_trade = min(
            trade["return_pct"]
            for trade in trades
        )

    else:

        win_rate = 0
        avg_trade = 0
        best_trade = 0
        worst_trade = 0

    return {
        "final_value": final_value,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "trades": len(trades)
    }