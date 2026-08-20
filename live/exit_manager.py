import argparse
import csv
import json
import os
import time

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient

from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
)

from alpaca.trading.enums import (
    QueryOrderStatus,
)


# ============================================================
# SAFETY SETTINGS
# ============================================================

# Leave True for the first test.
#
# True:
#   evaluates positions and prints hypothetical exits
#
# False:
#   submits exit orders to the Alpaca PAPER account
DRY_RUN = False

# Never change this while testing.
PAPER_MODE = True


# ============================================================
# EXIT RULES
# ============================================================

# Sell if position loses 3% from entry.
STOP_LOSS_PERCENT = -0.03

# Begin trailing the position after it has gained 3%.
TRAIL_ACTIVATION_PERCENT = 0.03

# Once trailing is active, sell after falling 2.5%
# from the highest observed price.
TRAIL_DISTANCE_PERCENT = 0.025

# Sell immediately if the position gains 8%.
HARD_TAKE_PROFIT_PERCENT = 0.08

# Force the position closed after five trading sessions.
MAX_HOLD_TRADING_DAYS = 5


# ============================================================
# LOOP SETTINGS
# ============================================================

CHECK_INTERVAL_SECONDS = 60


# ============================================================
# FILES
# ============================================================

STATE_PATH = "live/exit_state.json"

LOG_PATH = "live/exit_log.csv"


# ============================================================
# TIMEZONE
# ============================================================

NEW_YORK = ZoneInfo(
    "America/New_York"
)


# ============================================================
# ALPACA
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
        "Missing ALPACA_API_KEY or "
        "ALPACA_SECRET_KEY in .env"
    )


trading_client = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=PAPER_MODE,
)


# ============================================================
# LOGGING
# ============================================================

LOG_COLUMNS = [
    "timestamp",
    "symbol",
    "action",
    "reason",
    "entry_price",
    "current_price",
    "highest_price",
    "return_percent",
    "drawdown_from_high_percent",
    "trading_days_held",
    "quantity",
    "order_id",
    "dry_run",
    "details",
]


def append_log(**values):

    Path(LOG_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = os.path.exists(
        LOG_PATH
    )

    row = {
        column: values.get(
            column,
            "",
        )
        for column in LOG_COLUMNS
    }

    with open(
        LOG_PATH,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=LOG_COLUMNS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            row
        )


# ============================================================
# STATE STORAGE
# ============================================================

def load_state():

    if not os.path.exists(
        STATE_PATH
    ):

        return {}

    try:

        with open(
            STATE_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            result = json.load(
                file
            )

        if not isinstance(
            result,
            dict,
        ):

            return {}

        return result

    except Exception as error:

        print(
            f"Could not read exit state: "
            f"{error}"
        )

        return {}


def save_state(state):

    Path(STATE_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        STATE_PATH
        + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            state,
            file,
            indent=4,
        )

    os.replace(
        temporary_path,
        STATE_PATH,
    )


# ============================================================
# DATE UTILITIES
# ============================================================

def parse_datetime(value):

    if isinstance(
        value,
        datetime,
    ):

        result = value

    else:

        text = str(
            value
        ).replace(
            "Z",
            "+00:00",
        )

        result = datetime.fromisoformat(
            text
        )

    if result.tzinfo is None:

        result = result.replace(
            tzinfo=timezone.utc
        )

    return result


def count_trading_days_held(
    entry_time,
    current_time,
):

    entry_time = parse_datetime(
        entry_time
    )

    current_time = parse_datetime(
        current_time
    )

    entry_date = (
        entry_time
        .astimezone(
            NEW_YORK
        )
        .date()
    )

    current_date = (
        current_time
        .astimezone(
            NEW_YORK
        )
        .date()
    )

    if current_date <= entry_date:

        return 0

    request = GetCalendarRequest(
        start=entry_date,
        end=current_date,
    )

    calendar = trading_client.get_calendar(
        filters=request
    )

    if not calendar:

        return 0

    # Entry session counts as day zero.
    return max(
        0,
        len(calendar) - 1,
    )


# ============================================================
# ENTRY REGISTRATION
# ============================================================

def register_entry(
    symbol,
    entry_price,
    quantity,
    probability=None,
    source="unknown",
    entry_time=None,
):

    symbol = (
        str(symbol)
        .upper()
        .strip()
    )

    entry_price = float(
        entry_price
    )

    quantity = float(
        quantity
    )

    if entry_time is None:

        entry_time = datetime.now(
            timezone.utc
        )

    entry_time = parse_datetime(
        entry_time
    )

    state = load_state()

    state[symbol] = {
        "entry_time":
            entry_time.isoformat(),

        "entry_price":
            entry_price,

        "highest_price":
            entry_price,

        "quantity":
            quantity,

        "entry_score":
            (
                float(probability)
                if probability is not None
                else None
            ),

        "source":
            source,

        "exit_pending":
            False,

        "exit_order_id":
            None,

        "queued_exit_reason":
            None,

        "last_dry_run_reason":
            None,
    }

    save_state(
        state
    )

    print(
        f"\nExit manager registered "
        f"{symbol}:"
    )

    print(
        f"  Entry price: "
        f"${entry_price:,.2f}"
    )

    print(
        f"  Quantity: "
        f"{quantity:g}"
    )


# ============================================================
# ENUM HELPER
# ============================================================

def enum_value(value):

    if value is None:
        return ""

    if hasattr(
        value,
        "value",
    ):

        return str(
            value.value
        ).lower()

    return str(
        value
    ).lower()


# ============================================================
# OPEN ORDERS
# ============================================================

def get_open_orders(
    symbol,
):

    request = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol],
        limit=100,
    )

    return trading_client.get_orders(
        filter=request
    )


# ============================================================
# RECOVER UNREGISTERED POSITIONS
# ============================================================

def ensure_position_state(
    state,
    position,
    now,
):

    symbol = position.symbol

    average_entry_price = float(
        position.avg_entry_price
    )

    current_price = float(
        position.current_price
    )

    quantity = abs(
        float(
            position.qty
        )
    )

    if symbol in state:

        return state[symbol]

    # This happens when a position existed before the exit
    # manager was added. The five-day clock begins now because
    # we do not know the exact historical entry timestamp.
    state[symbol] = {
        "entry_time":
            now.isoformat(),

        "entry_price":
            average_entry_price,

        "highest_price":
            max(
                average_entry_price,
                current_price,
            ),

        "quantity":
            quantity,

        "entry_score":
            None,

        "source":
            "recovered_existing_position",

        "exit_pending":
            False,

        "exit_order_id":
            None,

        "queued_exit_reason":
            None,

        "last_dry_run_reason":
            None,
    }

    print(
        f"\nRecovered unregistered "
        f"position: {symbol}"
    )

    print(
        "Its maximum-hold clock starts now."
    )

    return state[symbol]


# ============================================================
# EXIT DECISION
# ============================================================

def determine_exit_reason(
    entry_price,
    current_price,
    highest_price,
    trading_days_held,
):

    return_percent = (
        current_price
        / entry_price
        - 1
    )

    highest_return = (
        highest_price
        / entry_price
        - 1
    )

    drawdown_from_high = (
        current_price
        / highest_price
        - 1
    )

    # Emergency loss control gets first priority.
    if (
        return_percent
        <= STOP_LOSS_PERCENT
    ):

        return "STOP_LOSS"

    # Lock in an unusually large gain.
    if (
        return_percent
        >= HARD_TAKE_PROFIT_PERCENT
    ):

        return "HARD_TAKE_PROFIT"

    # Trailing stop only becomes active after the stock
    # has first reached the activation profit.
    if (
        highest_return
        >= TRAIL_ACTIVATION_PERCENT
        and
        drawdown_from_high
        <= -TRAIL_DISTANCE_PERCENT
    ):

        return "TRAILING_STOP"

    if (
        trading_days_held
        >= MAX_HOLD_TRADING_DAYS
    ):

        return "MAX_HOLD_TIME"

    return None


# ============================================================
# PENDING EXIT ORDER
# ============================================================

def inspect_pending_exit(
    symbol,
    position_state,
):

    order_id = position_state.get(
        "exit_order_id"
    )

    if not order_id:

        position_state[
            "exit_pending"
        ] = False

        return False

    try:

        order = (
            trading_client
            .get_order_by_id(
                order_id
            )
        )

    except Exception as error:

        print(
            f"Could not inspect exit order "
            f"for {symbol}: {error}"
        )

        return True

    status = enum_value(
        order.status
    )

    if status in {
        "canceled",
        "cancelled",
        "rejected",
        "expired",
    }:

        print(
            f"{symbol} exit order "
            f"is {status}; it may retry."
        )

        position_state[
            "exit_pending"
        ] = False

        position_state[
            "exit_order_id"
        ] = None

        return False

    print(
        f"{symbol} exit order status: "
        f"{status}"
    )

    return True


# ============================================================
# SUBMIT EXIT
# ============================================================

def submit_exit(
    symbol,
    reason,
    position_state,
    metrics,
):

    open_orders = get_open_orders(
        symbol
    )

    if open_orders:

        print(
            f"Exit delayed: {symbol} "
            f"already has an open order."
        )

        append_log(
            timestamp=
                datetime.now(
                    timezone.utc
                ).isoformat(),

            symbol=symbol,
            action="EXIT_DELAYED",
            reason=reason,
            **metrics,

            order_id="",
            dry_run=DRY_RUN,
            details=
                "An open order already exists.",
        )

        return

    if DRY_RUN:

        if (
            position_state.get(
                "last_dry_run_reason"
            )
            != reason
        ):

            print(
                "\n=========================================="
            )

            print(
                "           DRY-RUN EXIT SIGNAL"
            )

            print(
                "=========================================="
            )

            print(
                f"\nWould close: "
                f"{symbol}"
            )

            print(
                f"Reason: "
                f"{reason}"
            )

            print(
                f"Return: "
                f"{metrics['return_percent']:.2%}"
            )

            print(
                f"Current price: "
                f"${metrics['current_price']:,.2f}"
            )

            print(
                "\nNO SELL ORDER WAS SUBMITTED."
            )

            append_log(
                timestamp=
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                symbol=symbol,
                action="DRY_RUN_EXIT",
                reason=reason,
                **metrics,

                order_id="",
                dry_run=True,
                details=
                    "No order submitted.",
            )

            position_state[
                "last_dry_run_reason"
            ] = reason

        return

    print(
        "\n=========================================="
    )

    print(
        "       SUBMITTING PAPER EXIT ORDER"
    )

    print(
        "=========================================="
    )

    print(
        f"\nSymbol: "
        f"{symbol}"
    )

    print(
        f"Reason: "
        f"{reason}"
    )

    # close_position liquidates the entire open position.
    order = trading_client.close_position(
        symbol
    )

    position_state[
        "exit_pending"
    ] = True

    position_state[
        "exit_order_id"
    ] = str(
        order.id
    )

    position_state[
        "queued_exit_reason"
    ] = None

    append_log(
        timestamp=
            datetime.now(
                timezone.utc
            ).isoformat(),

        symbol=symbol,
        action="EXIT_SUBMITTED",
        reason=reason,
        **metrics,

        order_id=str(
            order.id
        ),

        dry_run=False,
        details=
            "Full paper position close submitted.",
    )

    print(
        f"Exit order ID: "
        f"{order.id}"
    )


# ============================================================
# CLEAN CLOSED POSITIONS
# ============================================================

def clean_closed_positions(
    state,
    active_symbols,
):

    stale_symbols = [
        symbol
        for symbol in state
        if symbol not in active_symbols
    ]

    for symbol in stale_symbols:

        position_state = state[
            symbol
        ]

        print(
            f"\n{symbol} is no longer "
            f"an open position."
        )

        append_log(
            timestamp=
                datetime.now(
                    timezone.utc
                ).isoformat(),

            symbol=symbol,
            action="POSITION_CLOSED",
            reason=
                position_state.get(
                    "queued_exit_reason",
                    "",
                )
                or "",

            entry_price=
                position_state.get(
                    "entry_price",
                    "",
                ),

            current_price="",
            highest_price=
                position_state.get(
                    "highest_price",
                    "",
                ),

            return_percent="",
            drawdown_from_high_percent="",
            trading_days_held="",
            quantity=
                position_state.get(
                    "quantity",
                    "",
                ),

            order_id=
                position_state.get(
                    "exit_order_id",
                    "",
                )
                or "",

            dry_run=False,
            details=
                "Position no longer appears "
                "in Alpaca open positions.",
        )

        del state[
            symbol
        ]


# ============================================================
# CHECK ALL POSITIONS ONCE
# ============================================================

def check_positions_once():

    now = datetime.now(
        timezone.utc
    )

    clock = trading_client.get_clock()

    positions = (
        trading_client
        .get_all_positions()
    )

    state = load_state()

    active_symbols = {
        position.symbol
        for position in positions
    }

    clean_closed_positions(
        state,
        active_symbols,
    )

    print(
        "\n=========================================="
    )

    print(
        "             EXIT MANAGER"
    )

    print(
        "=========================================="
    )

    print(
        f"\nTime: "
        f"{clock.timestamp}"
    )

    print(
        f"Market open: "
        f"{clock.is_open}"
    )

    print(
        f"Paper mode: "
        f"{PAPER_MODE}"
    )

    print(
        f"Dry run: "
        f"{DRY_RUN}"
    )

    print(
        f"Open positions: "
        f"{len(positions)}"
    )

    if not positions:

        save_state(
            state
        )

        print(
            "\nNo positions to monitor."
        )

        return

    for position in positions:

        symbol = position.symbol

        side = enum_value(
            position.side
        )

        # This bot is currently designed for long positions.
        if "short" in side:

            print(
                f"\nSkipping short position "
                f"{symbol}."
            )

            continue

        position_state = ensure_position_state(
            state,
            position,
            now,
        )

        if position_state.get(
            "exit_pending",
            False,
        ):

            still_pending = inspect_pending_exit(
                symbol,
                position_state,
            )

            if still_pending:
                continue

        entry_price = float(
            position_state.get(
                "entry_price",
                position.avg_entry_price,
            )
        )

        current_price = float(
            position.current_price
        )

        quantity = abs(
            float(
                position.qty
            )
        )

        previous_high = float(
            position_state.get(
                "highest_price",
                entry_price,
            )
        )

        highest_price = max(
            previous_high,
            current_price,
        )

        position_state[
            "highest_price"
        ] = highest_price

        position_state[
            "quantity"
        ] = quantity

        trading_days_held = (
            count_trading_days_held(
                position_state[
                    "entry_time"
                ],
                now,
            )
        )

        return_percent = (
            current_price
            / entry_price
            - 1
        )

        drawdown_from_high = (
            current_price
            / highest_price
            - 1
        )

        trailing_active = (
            highest_price
            / entry_price
            - 1
            >= TRAIL_ACTIVATION_PERCENT
        )

        metrics = {
            "entry_price":
                entry_price,

            "current_price":
                current_price,

            "highest_price":
                highest_price,

            "return_percent":
                return_percent,

            "drawdown_from_high_percent":
                drawdown_from_high,

            "trading_days_held":
                trading_days_held,

            "quantity":
                quantity,
        }

        print(
            f"\n------------------------------------------"
        )

        print(
            f"{symbol}"
        )

        print(
            f"Entry:          "
            f"${entry_price:,.2f}"
        )

        print(
            f"Current:        "
            f"${current_price:,.2f}"
        )

        print(
            f"Highest seen:   "
            f"${highest_price:,.2f}"
        )

        print(
            f"Return:         "
            f"{return_percent:.2%}"
        )

        print(
            f"From high:      "
            f"{drawdown_from_high:.2%}"
        )

        print(
            f"Trading days:   "
            f"{trading_days_held}"
        )

        print(
            f"Trailing active:"
            f" {trailing_active}"
        )

        reason = position_state.get(
            "queued_exit_reason"
        )

        if not reason:

            reason = determine_exit_reason(
                entry_price=
                    entry_price,

                current_price=
                    current_price,

                highest_price=
                    highest_price,

                trading_days_held=
                    trading_days_held,
            )

        if not reason:

            position_state[
                "last_dry_run_reason"
            ] = None

            print(
                "Decision:        HOLD"
            )

            continue

        print(
            f"Decision:        SELL"
        )

        print(
            f"Reason:          "
            f"{reason}"
        )

        # Market exits are limited to the regular session in
        # this first version.
        if not clock.is_open:

            if DRY_RUN:

                print(
                    "Market closed: would queue "
                    "exit for next regular open."
                )

                submit_exit(
                    symbol,
                    reason,
                    position_state,
                    metrics,
                )

            else:

                position_state[
                    "queued_exit_reason"
                ] = reason

                print(
                    "Market closed: exit queued "
                    "for next regular open."
                )

            continue

        submit_exit(
            symbol,
            reason,
            position_state,
            metrics,
        )

    save_state(
        state
    )


# ============================================================
# MAIN LOOP
# ============================================================

def run_forever():

    print(
        "\n=========================================="
    )

    print(
        "       PAPER TRADING EXIT MANAGER"
    )

    print(
        "=========================================="
    )

    print(
        f"\nStop loss: "
        f"{STOP_LOSS_PERCENT:.1%}"
    )

    print(
        f"Trailing activation: "
        f"{TRAIL_ACTIVATION_PERCENT:.1%}"
    )

    print(
        f"Trailing distance: "
        f"{TRAIL_DISTANCE_PERCENT:.1%}"
    )

    print(
        f"Hard take profit: "
        f"{HARD_TAKE_PROFIT_PERCENT:.1%}"
    )

    print(
        f"Maximum hold: "
        f"{MAX_HOLD_TRADING_DAYS} "
        f"trading days"
    )

    print(
        f"Check interval: "
        f"{CHECK_INTERVAL_SECONDS} seconds"
    )

    print(
        "\nPress Ctrl+C to stop."
    )

    while True:

        try:

            check_positions_once()

        except KeyboardInterrupt:

            raise

        except Exception as error:

            print(
                "\nExit-manager error:"
            )

            print(error)

        time.sleep(
            CHECK_INTERVAL_SECONDS
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Monitor Alpaca paper positions "
            "and apply exit rules."
        )
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Check positions once and exit."
        ),
    )

    arguments = parser.parse_args()

    try:

        if arguments.once:

            check_positions_once()

        else:

            run_forever()

    except KeyboardInterrupt:

        print(
            "\nExit manager stopped."
        )