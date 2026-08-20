import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient


# ============================================================
# SETTINGS
# ============================================================

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

LOOKBACK_HOURS = 24
MAX_ARTICLES = 50

MAX_SYMBOLS_PER_ARTICLE = 4
MIN_IMPORTANCE = 2

# Historical/live articles that were edited days later can
# create misleading timing. We reject very stale updates.
MAX_UPDATE_AGE_HOURS = 24


# ============================================================
# API
# ============================================================

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        "Missing Alpaca API keys."
    )


HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}


trading_client = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True
)


asset_cache = {}


# ============================================================
# EVENT TYPES
# ============================================================

EVENT_TYPES = {

    "PRODUCT_LAUNCH": [
        "new product",
        "product launch",
        "launches new",
        "unveils",
        "unveiled",
        "new device",
        "new console",
        "game console",
        "gaming system",
        "game system",
        "preorders",
        "pre-orders",
    ],

    "EARNINGS": [
        "earnings",
        "quarterly results",
        "quarter results",
        "q1 eps",
        "q2 eps",
        "q3 eps",
        "q4 eps",
        "eps",
        "sales beat",
        "sales miss",
        "beats estimate",
        "misses estimate",
    ],

    "GUIDANCE": [
        "guidance",
        "outlook",
        "raises forecast",
        "raised forecast",
        "cuts forecast",
        "cut forecast",
        "raises guidance",
        "raised guidance",
        "cuts guidance",
        "cut guidance",
        "projected to",
        "expects to",
        "expected to",
        "reaffirms",
        "raises fy",
        "lowers fy",
    ],

    "CONTRACT": [
        "wins contract",
        "won contract",
        "awarded contract",
        "contract awarded",
        "selected by",
        "strategic partnership",
        "strategic agreement",
        "supply agreement",
        "deal with",
    ],

    "MERGER_ACQUISITION": [
        "acquires",
        "acquisition",
        "merger",
        "takeover",
        "buyout",
        "to acquire",
        "agrees to acquire",
    ],

    "REGULATORY": [
        "fda approval",
        "fda approves",
        "approved by fda",
        "regulatory approval",
        "regulatory rejection",
        "antitrust",
        "investigation",
        "lawsuit",
        "government approval",
    ],

    "ANALYST": [
        "upgrade",
        "upgrades",
        "downgrade",
        "downgrades",
        "price target",
        "initiates coverage",
        "analyst rating",
    ],
}


# Used when two event types receive equal scores.
EVENT_PRIORITY = {

    "MERGER_ACQUISITION": 7,
    "REGULATORY": 6,
    "GUIDANCE": 5,
    "PRODUCT_LAUNCH": 4,
    "CONTRACT": 3,
    "EARNINGS": 2,
    "ANALYST": 1,
}


# ============================================================
# SENTIMENT
# ============================================================

STRONG_POSITIVE = [
    "crushes estimates",
    "far above expectations",
    "raises guidance",
    "raises outlook",
    "record revenue",
    "record sales",
    "record demand",
    "sold out",
    "surging demand",
    "strong demand",
    "beats estimates",
    "beat estimates",
    "better than expected",
    "approval granted",
    "wins major contract",
]


POSITIVE = [
    "beat",
    "beats",
    "growth",
    "higher",
    "strong",
    "record",
    "raises",
    "raised",
    "approved",
    "approval",
    "exceeds",
    "surge",
]


STRONG_NEGATIVE = [
    "cuts guidance",
    "cuts outlook",
    "soft guidance",
    "far below expectations",
    "bankruptcy",
    "files for bankruptcy",
    "recall",
    "fraud",
    "regulatory rejection",
    "worse than expected",
]


NEGATIVE = [
    "miss",
    "misses",
    "weak",
    "lower",
    "decline",
    "cuts",
    "cut",
    "delay",
    "delayed",
    "lawsuit",
    "investigation",
    "ban",
    "falls",
    "drops",
]


# ============================================================
# FUTURE CATALYSTS
# ============================================================

FUTURE_PHRASES = [
    "will launch",
    "will release",
    "expected to launch",
    "expected to release",
    "plans to launch",
    "plans to release",
    "scheduled to",
    "set to launch",
    "set to release",
    "next month",
    "next quarter",
    "next week",
    "preorder",
    "pre-order",
    "upcoming",
    "coming soon",
    "expects demand",
    "expected demand",
    "projected to",
    "expects to",
    "expected to",
    "over next",
    "target by",
    "forecast",
]


# ============================================================
# POST-MOVE ARTICLES
# ============================================================

POST_MOVE_PHRASES = [
    "stock falls",
    "stock rises",
    "stock jumps",
    "stock surges",
    "stock drops",
    "shares fall",
    "shares rise",
    "shares jump",
    "shares surge",
    "shares drop",
    "stocks moving",
    "after-market session",
    "pre-market session",
    "why shares are trading",
    "why is the stock",
]


# ============================================================
# TIMESTAMP HANDLING
# ============================================================

def get_event_time(
    article,
    max_update_age_hours=MAX_UPDATE_AGE_HOURS
):

    created_raw = article.get(
        "created_at"
    )

    updated_raw = article.get(
        "updated_at"
    )


    if not created_raw:
        return None


    try:
        created_at = pd.to_datetime(
            created_raw,
            utc=True
        )

    except Exception:
        return None


    if updated_raw:

        try:
            updated_at = pd.to_datetime(
                updated_raw,
                utc=True
            )

        except Exception:
            updated_at = created_at

    else:
        updated_at = created_at


    # Protect against malformed timestamps.
    if updated_at < created_at:
        updated_at = created_at


    update_age_hours = (
        (
            updated_at
            - created_at
        ).total_seconds()
        / 3600
    )


    if (
        max_update_age_hours is not None
        and
        update_age_hours
        > max_update_age_hours
    ):
        return None


    event_time = max(
        created_at,
        updated_at
    )


    return {
        "created_at": created_at,
        "updated_at": updated_at,
        "event_time": event_time,
        "update_age_hours": update_age_hours,
    }


# ============================================================
# FETCH LIVE NEWS
# ============================================================

def fetch_news():

    end = datetime.now(
        timezone.utc
    )

    start = (
        end
        - timedelta(
            hours=LOOKBACK_HOURS
        )
    )


    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": MAX_ARTICLES,
        "sort": "desc",
        "include_content": "true",
    }


    response = requests.get(
        NEWS_URL,
        headers=HEADERS,
        params=params,
        timeout=20
    )


    response.raise_for_status()


    return response.json().get(
        "news",
        []
    )


# ============================================================
# SYMBOL FILTERING
# ============================================================

def is_tradable_us_equity(
    symbol
):

    if ":" in symbol:
        return False


    if symbol.endswith(
        "USD"
    ):
        return False


    if symbol in asset_cache:
        return asset_cache[
            symbol
        ]


    try:

        asset = trading_client.get_asset(
            symbol
        )


        tradable = bool(
            asset.tradable
        )


        asset_cache[
            symbol
        ] = tradable


        return tradable


    except Exception:

        asset_cache[
            symbol
        ] = False

        return False


def clean_symbols(
    symbols
):

    clean = []


    for symbol in symbols:

        symbol = str(
            symbol
        ).upper().strip()


        if not re.fullmatch(
            r"[A-Z]{1,5}",
            symbol
        ):
            continue


        if is_tradable_us_equity(
            symbol
        ):

            clean.append(
                symbol
            )


    return list(
        dict.fromkeys(
            clean
        )
    )


# ============================================================
# EVENT CLASSIFICATION
# ============================================================

def classify_event(
    headline,
    summary
):

    headline_text = (
        headline.lower()
    )

    combined = (
        headline
        + " "
        + summary
    ).lower()


    scores = {}


    for event_type, phrases in (
        EVENT_TYPES.items()
    ):

        score = 0


        for phrase in phrases:

            if phrase in headline_text:
                score += 3

            elif phrase in combined:
                score += 1


        if score > 0:

            scores[
                event_type
            ] = score


    if not scores:

        return (
            "OTHER",
            0
        )


    best_event = max(

        scores.keys(),

        key=lambda event: (
            scores[event],
            EVENT_PRIORITY.get(
                event,
                0
            )
        )
    )


    return (
        best_event,
        scores[
            best_event
        ]
    )


# ============================================================
# SENTIMENT
# ============================================================

def sentiment_score(
    headline,
    summary
):

    headline_lower = (
        headline.lower()
    )

    combined = (
        headline
        + " "
        + summary
    ).lower()


    score = 0


    for phrase in STRONG_POSITIVE:

        if phrase in headline_lower:
            score += 4

        elif phrase in combined:
            score += 2


    for phrase in STRONG_NEGATIVE:

        if phrase in headline_lower:
            score -= 4

        elif phrase in combined:
            score -= 2


    for phrase in POSITIVE:

        if phrase in headline_lower:
            score += 2

        elif phrase in combined:
            score += 1


    for phrase in NEGATIVE:

        if phrase in headline_lower:
            score -= 2

        elif phrase in combined:
            score -= 1


    return max(
        -10,
        min(
            10,
            score
        )
    )


# ============================================================
# FUTURE CATALYST
# ============================================================

def future_catalyst_score(
    headline,
    summary
):

    combined = (
        headline
        + " "
        + summary
    ).lower()


    score = 0


    for phrase in FUTURE_PHRASES:

        if phrase in combined:
            score += 1


    # Catch forward-looking year language.
    if re.search(
        r"\b20(?:2[6-9]|3[0-5])\b",
        combined
    ):

        score += 1


    return min(
        score,
        5
    )


# ============================================================
# POST-MOVE DETECTION
# ============================================================

def is_post_move_article(
    headline
):

    text = headline.lower()


    for phrase in POST_MOVE_PHRASES:

        if phrase in text:
            return True


    # Examples:
    # "Shares jump 18%"
    # "Stock down 12%"
    price_move_patterns = [

        r"\b(?:shares?|stock)\s+"
        r"(?:jump|jumps|surge|surges|rise|rises|"
        r"fall|falls|drop|drops|gain|gains|"
        r"plunge|plunges|soar|soars)"
        r"\s+\d+(?:\.\d+)?%",

        r"\b(?:shares?|stock)\s+"
        r"(?:up|down|higher|lower)"
        r"\s+\d+(?:\.\d+)?%",
    ]


    for pattern in price_move_patterns:

        if re.search(
            pattern,
            text
        ):

            return True


    return False


# ============================================================
# IMPORTANCE
# ============================================================

def calculate_importance(
    event_type,
    event_score,
    sentiment,
    future_score,
    post_move
):

    importance = 0


    importance += min(
        event_score,
        5
    )


    importance += min(
        abs(sentiment),
        5
    )


    importance += (
        future_score
        * 3
    )


    if event_type in [
        "PRODUCT_LAUNCH",
        "GUIDANCE",
        "CONTRACT",
        "MERGER_ACQUISITION",
        "REGULATORY",
    ]:

        importance += 2


    if post_move:

        importance -= 6


    return importance


# ============================================================
# PROCESS ARTICLE
# ============================================================

def process_article(
    article
):

    timing = get_event_time(
        article
    )


    if timing is None:
        return None


    headline = (
        article.get(
            "headline",
            ""
        )
        or ""
    )


    summary = (
        article.get(
            "summary",
            ""
        )
        or ""
    )


    raw_symbols = (
        article.get(
            "symbols",
            []
        )
        or []
    )


    symbols = clean_symbols(
        raw_symbols
    )


    if not symbols:
        return None


    if (
        len(symbols)
        > MAX_SYMBOLS_PER_ARTICLE
    ):
        return None


    event_type, event_score = (
        classify_event(
            headline,
            summary
        )
    )


    sentiment = sentiment_score(
        headline,
        summary
    )


    future_score = (
        future_catalyst_score(
            headline,
            summary
        )
    )


    post_move = (
        is_post_move_article(
            headline
        )
    )


    importance = (
        calculate_importance(
            event_type,
            event_score,
            sentiment,
            future_score,
            post_move
        )
    )


    return {

        "article_id":
            article.get(
                "id"
            ),

        "headline":
            headline,

        "summary":
            summary,

        "symbols":
            symbols,

        "event_type":
            event_type,

        "event_score":
            event_score,

        "sentiment":
            sentiment,

        "future_score":
            future_score,

        "post_move":
            post_move,

        "importance":
            importance,

        "created_at":
            timing[
                "created_at"
            ],

        "updated_at":
            timing[
                "updated_at"
            ],

        "event_time":
            timing[
                "event_time"
            ],

        "update_age_hours":
            timing[
                "update_age_hours"
            ],

        "source":
            article.get(
                "source",
                "Unknown"
            ),
    }


# ============================================================
# SCAN
# ============================================================

def scan_news():

    print(
        "\n=========================================="
    )

    print(
        "          V4 NEWS SCOUT - V3"
    )

    print(
        "=========================================="
    )


    print(
        f"\nScanning last "
        f"{LOOKBACK_HOURS} hours..."
    )


    articles = fetch_news()


    print(
        f"Articles received: "
        f"{len(articles)}"
    )


    results = []


    for article in articles:

        result = process_article(
            article
        )


        if result is None:
            continue


        if (
            result[
                "importance"
            ]
            < MIN_IMPORTANCE
        ):
            continue


        results.append(
            result
        )


    results.sort(

        key=lambda x:
            x[
                "importance"
            ],

        reverse=True
    )


    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = scan_news()


    print(
        "\n=========================================="
    )

    print(
        "         TOP US EQUITY CANDIDATES"
    )

    print(
        "=========================================="
    )


    if not results:

        print(
            "\nNo meaningful candidates found."
        )

        raise SystemExit


    for result in results[:20]:

        print(
            "\n------------------------------------------"
        )


        print(
            f"Symbols: "
            f"{', '.join(result['symbols'])}"
        )


        print(
            f"Event: "
            f"{result['event_type']}"
        )


        print(
            f"Sentiment: "
            f"{result['sentiment']}"
        )


        print(
            f"Future catalyst: "
            f"{result['future_score']}"
        )


        print(
            f"Post-move article: "
            f"{result['post_move']}"
        )


        print(
            f"Importance: "
            f"{result['importance']}"
        )


        print(
            f"Event time: "
            f"{result['event_time']}"
        )


        print(
            f"Update age: "
            f"{result['update_age_hours']:.2f}h"
        )


        print(
            "Headline:"
        )


        print(
            result[
                "headline"
            ]
        )