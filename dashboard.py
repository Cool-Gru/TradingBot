from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetCalendarRequest, GetOrdersRequest

load_dotenv()

# --------------------------- Settings ---------------------------
PAPER_MODE = True
REFRESH_SECONDS = 10
STARTING_BALANCE = float(os.getenv("BOT_STARTING_BALANCE", "100000"))

SIGNAL_LOG = Path("live/realtime_news_log.csv")
TRADE_LOG = Path("live/trade_log.csv")
EXIT_LOG = Path("live/exit_log.csv")
EXIT_STATE = Path("live/exit_state.json")
EQUITY_LOG = Path("live/dashboard_equity.csv")
OOS_FILE = Path("news/spike_oos_predictions_v5.csv")

STOP_LOSS = -0.03
TRAIL_ACTIVATION = 0.03
TRAIL_DISTANCE = 0.025
TAKE_PROFIT = 0.08
MAX_HOLD_DAYS = 5

BUY_DECISIONS = {
    "DRY_RUN_BUY",
    "PAPER_BUY_FILLED",
    "BUY_SIGNAL",
    "PREMARKET_BUY_WATCH",
    "NEXT_OPEN_BUY_WATCH",
    "STARTUP_BUY_WATCH",
}
WATCH_DECISIONS = {
    "PREMARKET_BUY_WATCH",
    "NEXT_OPEN_BUY_WATCH",
    "STARTUP_BUY_WATCH",
}

st.set_page_config(
    page_title="AI Trading Bot Command Center",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .banner {padding: .75rem; border: 1px solid rgba(128,128,128,.35);
               border-radius: .7rem; font-weight: 800; text-align: center;}
      .card {padding: 1rem; border: 1px solid rgba(128,128,128,.35);
             border-radius: .7rem; margin-bottom: .8rem;}
      .buy {border-left: .45rem solid #2ca02c;}
      .sell {border-left: .45rem solid #d62728;}
      .watch {border-left: .45rem solid #f2a900;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2);
                                  padding: .65rem; border-radius: .6rem;}
      div[data-testid="stMetricValue"] {font-size: clamp(1.35rem, 1.8vw, 2.1rem);}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Helpers ---------------------------
def num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def money(value: Any) -> str:
    return f"${num(value):,.2f}"


def pct(value: Any) -> str:
    return f"{num(value):.2%}"


def enum_text(value: Any) -> str:
    return str(getattr(value, "value", value or ""))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def as_et(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        ts = pd.Timestamp(value)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        return ts.tz_convert("America/New_York")
    except Exception:
        return None


@st.cache_resource
def trading_client() -> TradingClient:
    load_dotenv()
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env")
    return TradingClient(key, secret, paper=PAPER_MODE)


def process_running(module: str) -> bool | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return any(
            module.lower() in " ".join(p.info.get("cmdline") or []).lower()
            for p in psutil.process_iter(["cmdline"])
        )
    except Exception:
        return None


def append_equity(equity: float, cash: float) -> None:
    EQUITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    old = read_csv(EQUITY_LOG)
    if not old.empty and "timestamp" in old:
        last = pd.to_datetime(old["timestamp"].iloc[-1], utc=True, errors="coerce")
        if pd.notna(last) and (pd.Timestamp.now(tz="UTC") - last).total_seconds() < 60:
            return
    pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "equity": equity,
        "cash": cash,
    }]).to_csv(EQUITY_LOG, mode="a", header=not EQUITY_LOG.exists(), index=False)

# ----------------------- Validation stats -----------------------
def find_column(df: pd.DataFrame, choices: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    return next((lookup[c.lower()] for c in choices if c.lower() in lookup), None)


@st.cache_data(ttl=60)
def oos_stats() -> dict[str, Any]:
    df = read_csv(OOS_FILE)
    if df.empty:
        return {"available": False}

    pcol = find_column(df, [
        "probability", "predicted_probability", "score", "spike_score",
        "oos_probability", "pred",
    ])
    ycol = find_column(df, [
        "spike_5pct", "actual", "target", "label", "y_true", "outcome",
    ])
    if not pcol or not ycol:
        return {"available": False, "columns": list(df.columns)}

    p = pd.to_numeric(df[pcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    mask = p.notna() & y.notna()
    p, y = p[mask], y[mask].astype(int)

    thresholds = {}
    for threshold in (0.60, 0.65, 0.70, 0.75, 0.80):
        selected = p >= threshold
        count = int(selected.sum())
        thresholds[threshold] = {
            "signals": count,
            "precision": float(y[selected].mean()) if count else None,
        }
    return {
        "available": True,
        "rows": len(p),
        "baseline": float(y.mean()),
        "thresholds": thresholds,
    }


def nearest_validation(score: float, stats: dict[str, Any]) -> dict[str, Any] | None:
    if not stats.get("available"):
        return None
    candidates = [
        threshold for threshold, result in stats["thresholds"].items()
        if threshold <= score and result["signals"] > 0
    ]
    if not candidates:
        return None
    threshold = max(candidates)
    result = dict(stats["thresholds"][threshold])
    result["threshold"] = threshold
    return result

# ---------------------- Buy recommendations ----------------------
def signal_frame() -> pd.DataFrame:
    df = read_csv(SIGNAL_LOG)
    if df.empty:
        return df
    for col in [
        "spike_score", "sentiment", "importance", "current_price",
        "previous_close", "move_from_previous_close", "spread_percent",
        "avg_dollar_volume_20",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "processed_at" in df:
        df["_time"] = pd.to_datetime(df["processed_at"], utc=True, errors="coerce")
        df = df.sort_values("_time", ascending=False)
    if "article_time" in df:
        df["_article_time"] = pd.to_datetime(df["article_time"], utc=True, errors="coerce")
    if "expires_at" in df:
        df["_expires_at"] = pd.to_datetime(df["expires_at"], utc=True, errors="coerce")
    return df


def latest_buy(df: pd.DataFrame) -> pd.Series | None:
    if df.empty or "decision" not in df:
        return None
    good = df[df["decision"].astype(str).str.upper().isin(BUY_DECISIONS)]
    now = pd.Timestamp.now(tz="UTC")
    if "_expires_at" in good:
        has_expiration = good["_expires_at"].notna()
        fallback_fresh = (
            good["_time"] >= now - pd.Timedelta(hours=24)
            if "_time" in good
            else False
        )
        good = good[
            (has_expiration & (good["_expires_at"] >= now))
            | (~has_expiration & fallback_fresh)
        ]
    elif "_time" in good:
        good = good[good["_time"] >= now - pd.Timedelta(hours=24)]
    if good.empty:
        return None

    sort_columns = [column for column in ["spike_score", "_time"] if column in good]
    if sort_columns:
        good = good.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    return good.iloc[0]


def buy_rating(row: pd.Series) -> int:
    score = num(row.get("spike_score"))
    sentiment = num(row.get("sentiment"))
    importance = num(row.get("importance"))
    spread = num(row.get("spread_percent"), 1.0)
    move = num(row.get("move_from_previous_close"))

    points = 1 + max(0, min(6, (score - 0.55) / 0.05))
    points += 1.5 if sentiment >= 3 else 1 if sentiment >= 2 else 0
    points += 1 if importance >= 8 else 0.5 if importance >= 6 else 0
    points += 0.75 if spread <= 0.005 else 0.35 if spread <= 0.01 else 0
    points += 0.75 if move <= 0.01 else 0.35 if move <= 0.03 else -1.5 if move > 0.04 else 0
    return int(max(1, min(10, round(points))))

# ---------------------- Sell recommendations ----------------------
def held_days(client: TradingClient, entry_time: Any) -> int:
    ts = as_et(entry_time)
    if ts is None or date.today() <= ts.date():
        return 0
    try:
        sessions = client.get_calendar(filters=GetCalendarRequest(start=ts.date(), end=date.today()))
        return max(0, len(sessions) - 1)
    except Exception:
        return max(0, len(pd.bdate_range(ts.date(), date.today())) - 1)


def exit_reason(entry: float, current: float, high: float, days: int) -> str | None:
    if entry <= 0 or high <= 0:
        return None
    ret = current / entry - 1
    high_ret = high / entry - 1
    drawdown = current / high - 1
    if ret <= STOP_LOSS:
        return "STOP LOSS"
    if ret >= TAKE_PROFIT:
        return "TAKE PROFIT"
    if high_ret >= TRAIL_ACTIVATION and drawdown <= -TRAIL_DISTANCE:
        return "TRAILING STOP"
    if days >= MAX_HOLD_DAYS:
        return "MAX HOLD TIME"
    return None


def hold_action(ret: float, drawdown: float, days: int, reason: str | None) -> tuple[str, int, str]:
    if reason:
        return "SELL", 10, reason
    pressure = 1
    pressure = max(pressure, 8 if ret <= -0.02 else 6 if ret <= -0.01 else 1)
    pressure = max(pressure, 8 if drawdown <= -0.02 else 6 if drawdown <= -0.0125 else 1)
    pressure = max(pressure, 8 if ret >= 0.07 else 6 if ret >= 0.05 else 1)
    pressure = max(pressure, 8 if days >= 4 else 6 if days >= 3 else 1)
    if pressure >= 7:
        return "WATCH", pressure, "An exit rule is getting close."
    return "HOLD", max(1, 10 - pressure), "No configured exit rule has triggered."


def position_rows(client: TradingClient, positions: list[Any]) -> list[dict[str, Any]]:
    state = read_json(EXIT_STATE)
    rows = []
    for position in positions:
        symbol = str(position.symbol)
        stored = state.get(symbol, {})
        entry = num(stored.get("entry_price", position.avg_entry_price))
        current = num(position.current_price)
        high = max(entry, current, num(stored.get("highest_price")))
        ret = current / entry - 1 if entry else 0
        drawdown = current / high - 1 if high else 0
        days = held_days(client, stored.get("entry_time"))
        reason = exit_reason(entry, current, high, days)
        action, rating, explanation = hold_action(ret, drawdown, days, reason)
        rows.append({
            "symbol": symbol,
            "action": action,
            "rating": rating,
            "reason": explanation,
            "quantity": abs(num(position.qty)),
            "entry": entry,
            "current": current,
            "market_value": num(position.market_value),
            "unrealized_pl": num(position.unrealized_pl),
            "return": ret,
            "high": high,
            "drawdown": drawdown,
            "days": days,
            "stop_price": entry * (1 + STOP_LOSS),
            "trail_activation": entry * (1 + TRAIL_ACTIVATION),
            "take_profit": entry * (1 + TAKE_PROFIT),
            "trailing_active": (high / entry - 1 >= TRAIL_ACTIVATION) if entry else False,
            "entry_score": num(stored.get("entry_score")),
            "source": stored.get("source", "unknown"),
        })
    return rows

# --------------------------- UI pieces ---------------------------
def render_buy(recommendation: pd.Series | None, stats: dict[str, Any], held_symbols: set[str]) -> None:
    st.subheader("Current buy recommendation")
    if recommendation is None:
        st.markdown(
            '<div class="card watch"><h3>NO QUALIFIED BUY SIGNAL</h3>'
            '<p>No recent signal passed every model and market filter.</p></div>',
            unsafe_allow_html=True,
        )
        return

    symbol = str(recommendation.get("symbol", "UNKNOWN")).upper()
    score = num(recommendation.get("spike_score"))
    rating = buy_rating(recommendation)
    decision = str(recommendation.get("decision", "")).upper()
    is_watch = decision in WATCH_DECISIONS
    already_held = symbol in held_symbols
    action = "HOLD" if already_held else "BUY WATCH" if is_watch else "BUY"
    if already_held:
        note = "The entry signal remains strong, but this position is already open."
    elif is_watch:
        note = (
            "Recovered when the news model started. No delayed order was submitted; "
            "verify the quote and spread before the stated entry window expires."
        )
    else:
        note = "Paper-trading model signal; not a guaranteed return."
    st.markdown(
        f'<div class="card buy"><h2>{action} {symbol} — {rating}/10 MODEL RECOMMENDATION</h2>'
        f'<p>{note}</p></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    cols[0].metric("V5 score", pct(score))
    cols[1].metric("Sentiment", f"{num(recommendation.get('sentiment')):g}")
    cols[2].metric("Importance", f"{num(recommendation.get('importance')):g}")
    cols[3].metric("Current move", pct(recommendation.get("move_from_previous_close")))
    cols[4].metric("Spread", pct(recommendation.get("spread_percent")))

    st.markdown("#### Evidence")
    st.write(f"**Headline:** {recommendation.get('headline', '')}")
    st.write(f"**Event:** {recommendation.get('event_type', 'UNKNOWN')}")
    st.write(f"**Bot decision:** {recommendation.get('decision', '')}")
    article_time = as_et(recommendation.get("article_time"))
    expires_at = as_et(recommendation.get("expires_at"))
    if article_time is not None:
        st.write(f"**Article time:** {article_time:%Y-%m-%d %I:%M:%S %p ET}")
    if expires_at is not None:
        st.write(f"**Recommendation expires:** {expires_at:%Y-%m-%d %I:%M %p ET}")
    if recommendation.get("details"):
        st.write(f"**Status:** {recommendation.get('details')}")
    st.write(f"**20-day average dollar volume:** {money(recommendation.get('avg_dollar_volume_20'))}")

    validation = nearest_validation(score, stats)
    if validation and validation["precision"] is not None:
        st.info(
            f"Historical V5 out-of-sample signals at or above {validation['threshold']:.0%} "
            f"had {validation['precision']:.2%} spike precision across "
            f"{validation['signals']:,} signals."
        )
    st.warning(
        "V5 was validated for next-session entries. Immediate news execution is a different "
        "strategy, so this historical statistic is evidence—not a proven instant-trade win rate."
    )


def render_exits(rows: list[dict[str, Any]]) -> None:
    st.subheader("Sell and hold recommendations")
    if not rows:
        st.info("No open positions to evaluate.")
        return

    for row in rows:
        css = "sell" if row["action"] == "SELL" else "watch" if row["action"] == "WATCH" else "buy"
        noun = "EXIT SIGNAL" if row["action"] == "SELL" else "EXIT PRESSURE" if row["action"] == "WATCH" else "HOLD SIGNAL"
        st.markdown(
            f'<div class="card {css}"><h3>{row["action"]} {row["symbol"]} — '
            f'{row["rating"]}/10 {noun}</h3><p>{row["reason"]}</p></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(6)
        cols[0].metric("Return", pct(row["return"]), money(row["unrealized_pl"]))
        cols[1].metric("Current", money(row["current"]))
        cols[2].metric("Highest", money(row["high"]))
        cols[3].metric("From high", pct(row["drawdown"]))
        cols[4].metric("Days held", str(row["days"]))
        cols[5].metric("Trailing active", "YES" if row["trailing_active"] else "NO")
        with st.expander(f"{row['symbol']} thresholds and proof"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Stop-loss price", money(row["stop_price"]))
            c2.metric("Trailing activation", money(row["trail_activation"]))
            c3.metric("Take-profit price", money(row["take_profit"]))
            st.write(f"Shares: **{row['quantity']:g}**")
            st.write(f"Entry: **{money(row['entry'])}**")
            st.write(f"Entry source: **{row['source']}**")
            if row["entry_score"]:
                st.write(f"Entry model score: **{row['entry_score']:.2%}**")


def orders_dataframe(orders: list[Any]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Submitted": as_et(getattr(order, "submitted_at", None)),
        "Symbol": getattr(order, "symbol", ""),
        "Side": enum_text(getattr(order, "side", "")).upper(),
        "Type": enum_text(getattr(order, "type", "")).upper(),
        "Quantity": num(getattr(order, "qty", 0)),
        "Limit": num(getattr(order, "limit_price", 0)),
        "Filled": num(getattr(order, "filled_qty", 0)),
        "Fill price": num(getattr(order, "filled_avg_price", 0)),
        "Status": enum_text(getattr(order, "status", "")).upper(),
    } for order in orders])


def system_status(label: str, state: bool | None) -> None:
    if state is True:
        st.success(f"{label}: RUNNING")
    elif state is False:
        st.error(f"{label}: NOT DETECTED")
    else:
        st.info(f"{label}: UNKNOWN (install psutil)")

# --------------------------- Main app ---------------------------
st.title("AI Trading Bot Command Center")
st.caption(
    "Account balance, recommendations, evidence, positions, orders, exits, and system health."
)

with st.sidebar:
    st.header("Safety")
    st.success("Broker: ALPACA PAPER")
    st.warning(
        "The 1–10 ratings summarize your model and risk rules. They are not calibrated "
        "probabilities or guaranteed financial advice."
    )
    st.markdown("#### Exit rules")
    st.write(f"Stop loss: **{STOP_LOSS:.1%}**")
    st.write(f"Trailing activation: **{TRAIL_ACTIVATION:.1%}**")
    st.write(f"Trailing distance: **{TRAIL_DISTANCE:.1%}**")
    st.write(f"Take profit: **{TAKE_PROFIT:.1%}**")
    st.write(f"Maximum hold: **{MAX_HOLD_DAYS} sessions**")
    st.markdown("#### P&L baseline")
    st.write(f"Starting balance: **{money(STARTING_BALANCE)}**")
    st.caption("Set BOT_STARTING_BALANCE in .env if needed.")
    if st.button("Refresh now", use_container_width=True):
        st.rerun()


@st.fragment(run_every=f"{REFRESH_SECONDS}s")
def live_dashboard() -> None:
    try:
        client = trading_client()
        account = client.get_account()
        clock = client.get_clock()
        positions = list(client.get_all_positions())
        orders = list(client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100)))
    except Exception as exc:
        st.error("Could not connect to Alpaca.")
        st.exception(exc)
        return

    equity = num(account.equity)
    last_equity = num(getattr(account, "last_equity", equity), equity)
    cash = num(account.cash)
    buying_power = num(account.buying_power)
    day_pl = equity - last_equity
    total_pl = equity - STARTING_BALANCE
    append_equity(equity, cash)

    st.markdown(
        f'<div class="banner">ALPACA PAPER • MARKET {"OPEN" if clock.is_open else "CLOSED"} '
        f'• AUTO-REFRESH {REFRESH_SECONDS}s</div>',
        unsafe_allow_html=True,
    )

    account_metrics = st.columns(4)
    account_metrics[0].metric("Equity", money(equity))
    account_metrics[1].metric("Cash", money(cash))
    account_metrics[2].metric("Buying power", money(buying_power))
    account_metrics[3].metric(
        "Today P&L",
        money(day_pl),
        pct(day_pl / last_equity if last_equity else 0),
    )

    status_metrics = st.columns(3)
    status_metrics[0].metric(
        "Total P&L",
        money(total_pl),
        pct(total_pl / STARTING_BALANCE if STARTING_BALANCE else 0),
    )
    status_metrics[1].metric("Positions", len(positions))
    status_metrics[2].metric("Market", "OPEN" if clock.is_open else "CLOSED")

    signals = signal_frame()
    rows = position_rows(client, positions)
    stats = oos_stats()

    left, right = st.columns([1.05, 1.35])
    with left:
        render_buy(latest_buy(signals), stats, {str(p.symbol).upper() for p in positions})
    with right:
        render_exits(rows)

    tabs = st.tabs(["Positions", "News & signals", "Orders", "Equity", "Performance", "System health"])

    with tabs[0]:
        if not rows:
            st.info("No open positions.")
        else:
            table = pd.DataFrame(rows)[[
                "symbol", "action", "rating", "quantity", "entry", "current",
                "market_value", "unrealized_pl", "return", "high", "drawdown",
                "days", "trailing_active",
            ]]
            st.dataframe(table, use_container_width=True, hide_index=True)

    with tabs[1]:
        if signals.empty:
            st.info(f"No signal log found at {SIGNAL_LOG}.")
        else:
            preferred = [
                "processed_at", "symbol", "headline", "event_type", "sentiment",
                "importance", "spike_score", "move_from_previous_close",
                "spread_percent", "decision", "details", "source_mode", "expires_at",
            ]
            st.dataframe(signals[[c for c in preferred if c in signals]].head(50), use_container_width=True, hide_index=True)

    with tabs[2]:
        table = orders_dataframe(orders)
        st.info("No orders found.") if table.empty else st.dataframe(table, use_container_width=True, hide_index=True)

    with tabs[3]:
        history = read_csv(EQUITY_LOG)
        if history.empty:
            st.info("The equity chart will appear after two snapshots.")
        else:
            history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
            history["equity"] = pd.to_numeric(history["equity"], errors="coerce")
            history = history.dropna(subset=["timestamp", "equity"])
            if len(history) < 2:
                st.info("Waiting for another snapshot.")
            else:
                st.line_chart(history.set_index("timestamp")[["equity"]], use_container_width=True)

    with tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Trade log")
            df = read_csv(TRADE_LOG)
            st.info(f"No data at {TRADE_LOG}.") if df.empty else st.dataframe(df.tail(50), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Exit log")
            df = read_csv(EXIT_LOG)
            st.info(f"No data at {EXIT_LOG}.") if df.empty else st.dataframe(df.tail(50), use_container_width=True, hide_index=True)

    with tabs[5]:
        c1, c2, c3 = st.columns(3)
        with c1:
            system_status("News stream", process_running("live.news_stream"))
        with c2:
            system_status("Exit manager", process_running("live.exit_manager"))
        with c3:
            st.success("Alpaca API: CONNECTED")
        st.write(f"**Alpaca clock:** {clock.timestamp}")
        if not stats.get("available"):
            st.warning(
                "V5 OOS statistics could not be loaded. The dashboard will still work, "
                "but historical proof will not be shown."
            )


live_dashboard()
