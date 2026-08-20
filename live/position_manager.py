import os
import csv
from datetime import datetime, timezone

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# ============================================================
# SETTINGS
# ============================================================

HOLD_DAYS = 5

DRY_RUN = True

TRADE_LOG_PATH = "live/trade_log.csv"


# ============================================================
# ALPACA CONNECTION
# ============================================================

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


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
# LOG FILE
# ============================================================

LOG_FIELDS = [
    "symbol",
    "entry_date",
    "entry_price",
    "quantity",
    "ai_probability",
    "status",
    "exit_date",
    "exit_price",
    "profit_loss",
]


def ensure_log_exists():

    if os.path.exists(
        TRADE_LOG_PATH
    ):
        return


    with open(
        TRADE_LOG_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=LOG_FIELDS
        )

        writer.writeheader()


# ============================================================
# READ LOG
# ============================================================

def read_trade_log():

    ensure_log_exists()

    rows = []

    with open(
        TRADE_LOG_PATH,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# WRITE COMPLETE LOG
# ============================================================

def write_trade_log(rows):

    with open(
        TRADE_LOG_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=LOG_FIELDS
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# RECORD ENTRY
# ============================================================

def record_entry(
    symbol,
    entry_price,
    quantity,
    probability
):

    ensure_log_exists()


    row = {
        "symbol":
            symbol,

        "entry_date":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "entry_price":
            entry_price,

        "quantity":
            quantity,

        "ai_probability":
            probability,

        "status":
            "OPEN",

        "exit_date":
            "",

        "exit_price":
            "",

        "profit_loss":
            "",
    }


    with open(
        TRADE_LOG_PATH,
        "a",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=LOG_FIELDS
        )

        writer.writerow(
            row
        )


# ============================================================
# FIND OPEN LOGGED TRADE
# ============================================================

def get_open_logged_trade():

    rows = read_trade_log()

    for row in reversed(rows):

        if row["status"] == "OPEN":

            return row

    return None


# ============================================================
# TRADING DAYS HELD
# ============================================================

def trading_days_held(
    entry_date_string
):

    entry_date = datetime.fromisoformat(
        entry_date_string
    )

    now = datetime.now(
        timezone.utc
    )


    # Basic weekday count.
    #
    # Good enough for first paper version.
    # Later we'll replace this with an
    # exchange-calendar-aware counter.

    days = 0

    current = entry_date.date()

    end_date = now.date()


    while current < end_date:

        current = (
            current
            .fromordinal(
                current.toordinal() + 1
            )
        )

        if current.weekday() < 5:

            days += 1


    return days


# ============================================================
# FIND ACTUAL ALPACA POSITION
# ============================================================

def find_position(symbol):

    positions = (
        trading_client
        .get_all_positions()
    )


    for position in positions:

        if position.symbol == symbol:

            return position


    return None


# ============================================================
# CLOSE LOG RECORD
# ============================================================

def mark_trade_closed(
    symbol,
    exit_price,
    profit_loss
):

    rows = read_trade_log()


    for row in reversed(rows):

        if (
            row["symbol"] == symbol
            and
            row["status"] == "OPEN"
        ):

            row["status"] = "CLOSED"

            row["exit_date"] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            row["exit_price"] = (
                exit_price
            )

            row["profit_loss"] = (
                profit_loss
            )

            break


    write_trade_log(
        rows
    )


# ============================================================
# CHECK FOR EXIT
# ============================================================

def check_position_exit():

    logged_trade = (
        get_open_logged_trade()
    )


    if logged_trade is None:

        return {
            "action":
                "NONE",

            "message":
                "No logged open trade."
        }


    symbol = (
        logged_trade[
            "symbol"
        ]
    )


    held_days = trading_days_held(
        logged_trade[
            "entry_date"
        ]
    )


    position = find_position(
        symbol
    )


    if position is None:

        return {
            "action":
                "WARNING",

            "message":
                (
                    f"{symbol} is logged OPEN "
                    "but Alpaca has no position."
                )
        }


    current_price = float(
        position.current_price
    )

    entry_price = float(
        logged_trade[
            "entry_price"
        ]
    )

    quantity = float(
        logged_trade[
            "quantity"
        ]
    )


    profit_loss = (
        current_price
        - entry_price
    ) * quantity


    print("\n===================================")
    print("        POSITION MANAGER")
    print("===================================")

    print(
        f"\nSymbol:        "
        f"{symbol}"
    )

    print(
        f"Held days:     "
        f"{held_days}"
    )

    print(
        f"Entry price:   "
        f"${entry_price:.2f}"
    )

    print(
        f"Current price: "
        f"${current_price:.2f}"
    )

    print(
        f"Current P/L:   "
        f"${profit_loss:.2f}"
    )


    if held_days < HOLD_DAYS:

        return {
            "action":
                "HOLD",

            "message":
                (
                    f"Hold {symbol}. "
                    f"{HOLD_DAYS - held_days} "
                    "trading days remaining."
                )
        }


    # ========================================================
    # EXIT DUE
    # ========================================================

    qty = float(
        position.qty
    )


    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )


    if DRY_RUN:

        return {
            "action":
                "DRY_RUN_SELL",

            "message":
                (
                    f"Would sell {qty} shares "
                    f"of {symbol}."
                )
        }


    clock = trading_client.get_clock()


    if not clock.is_open:

        return {
            "action":
                "WAIT",

            "message":
                "Exit is due, but market is closed."
        }


    order = (
        trading_client
        .submit_order(
            order_data=order_request
        )
    )


    mark_trade_closed(
        symbol,
        current_price,
        profit_loss
    )


    return {
        "action":
            "SELL",

        "message":
            (
                f"Submitted SELL for "
                f"{qty} {symbol}."
            ),

        "order_id":
            str(order.id)
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    ensure_log_exists()

    result = check_position_exit()

    print(
        "\nResult:"
    )

    print(
        result["message"]
    )