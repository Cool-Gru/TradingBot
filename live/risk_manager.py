import os

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus


# ============================================================
# SETTINGS
# ============================================================

MAX_POSITION_PERCENT = 0.20

MAX_OPEN_POSITIONS = 1

MIN_CASH = 100.00


# ============================================================
# API
# ============================================================

load_dotenv()

API_KEY = os.getenv(
    "ALPACA_API_KEY"
)

SECRET_KEY = os.getenv(
    "ALPACA_SECRET_KEY"
)


if not API_KEY or not SECRET_KEY:

    raise RuntimeError(
        "Missing Alpaca API keys."
    )


trading_client = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True
)


# ============================================================
# ACCOUNT
# ============================================================

def get_account_status():

    account = (
        trading_client
        .get_account()
    )


    return {

        "equity":
            float(
                account.equity
            ),

        "cash":
            float(
                account.cash
            ),

        "buying_power":
            float(
                account.buying_power
            ),

        "trading_blocked":
            bool(
                getattr(
                    account,
                    "trading_blocked",
                    False
                )
            ),

        "account_blocked":
            bool(
                getattr(
                    account,
                    "account_blocked",
                    False
                )
            ),

        "trade_suspended":
            bool(
                getattr(
                    account,
                    "trade_suspended_by_user",
                    False
                )
            ),
    }


# ============================================================
# POSITIONS
# ============================================================

def get_open_positions():

    return (
        trading_client
        .get_all_positions()
    )


# ============================================================
# OPEN ORDERS
# ============================================================

def get_open_orders(
    symbol=None
):

    symbols = None


    if symbol:

        symbols = [
            symbol
        ]


    request = GetOrdersRequest(

        status=
            QueryOrderStatus.OPEN,

        limit=100,

        symbols=symbols
    )


    return (
        trading_client
        .get_orders(
            filter=request
        )
    )


# ============================================================
# BUDGET
# ============================================================

def calculate_position_budget():

    account = (
        get_account_status()
    )


    position_limit = (

        account[
            "equity"
        ]

        * MAX_POSITION_PERCENT
    )


    # IMPORTANT:
    #
    # Cash is included here so we do not
    # intentionally use margin.
    budget = min(

        position_limit,

        account[
            "cash"
        ],

        account[
            "buying_power"
        ]
    )


    return max(
        0.0,
        budget
    )


# ============================================================
# TRADE APPROVAL
# ============================================================

def can_open_trade(
    symbol
):

    symbol = (
        symbol
        .upper()
        .strip()
    )


    account = (
        get_account_status()
    )


    positions = (
        get_open_positions()
    )


    open_orders = (
        get_open_orders(
            symbol
        )
    )


    reasons = []


    if account[
        "trading_blocked"
    ]:

        reasons.append(
            "Account trading is blocked."
        )


    if account[
        "account_blocked"
    ]:

        reasons.append(
            "Account is blocked."
        )


    if account[
        "trade_suspended"
    ]:

        reasons.append(
            "Trading is suspended."
        )


    if account[
        "cash"
    ] < MIN_CASH:

        reasons.append(
            "Not enough available cash."
        )


    if len(
        positions
    ) >= MAX_OPEN_POSITIONS:

        reasons.append(
            "Maximum open positions reached."
        )


    for position in positions:

        if (
            position.symbol
            == symbol
        ):

            reasons.append(
                f"Already holding {symbol}."
            )


    if len(
        open_orders
    ) > 0:

        reasons.append(
            f"An open order already exists for "
            f"{symbol}."
        )


    budget = (
        calculate_position_budget()
    )


    if budget <= 0:

        reasons.append(
            "Trade budget is zero."
        )


    return {

        "approved":
            len(
                reasons
            ) == 0,

        "reasons":
            reasons,

        "account":
            account,

        "positions":
            positions,

        "open_orders":
            open_orders,

        "budget":
            budget,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n==================================="
    )

    print(
        "         RISK MANAGER TEST"
    )

    print(
        "==================================="
    )


    account = (
        get_account_status()
    )


    positions = (
        get_open_positions()
    )


    budget = (
        calculate_position_budget()
    )


    print(
        f"\nEquity:       "
        f"${account['equity']:,.2f}"
    )


    print(
        f"Cash:         "
        f"${account['cash']:,.2f}"
    )


    print(
        f"Buying Power: "
        f"${account['buying_power']:,.2f}"
    )


    print(
        f"\nOpen positions: "
        f"{len(positions)}"
    )


    print(
        f"Maximum trade budget: "
        f"${budget:,.2f}"
    )


    result = (
        can_open_trade(
            "AAPL"
        )
    )


    print(
        f"\nCan trade AAPL: "
        f"{result['approved']}"
    )


    if result[
        "reasons"
    ]:

        print(
            "\nReasons:"
        )

        for reason in result[
            "reasons"
        ]:

            print(
                f" - {reason}"
            )