AI Trading Bot Dashboard
========================

Copy these files into C:\FunStuff\TradingBot:
- dashboard.py
- dashboard_requirements.txt
- run_dashboard.ps1

Install:
    .\.venv\Scripts\Activate.ps1
    pip install -r dashboard_requirements.txt

Run:
    python -m streamlit run dashboard.py

Or:
    .\run_dashboard.ps1

Optional .env setting:
    BOT_STARTING_BALANCE=100000

The dashboard reads:
- Alpaca paper account, positions, orders, and market clock
- live/realtime_news_log.csv
- live/exit_state.json
- live/exit_log.csv
- live/trade_log.csv
- news/spike_oos_predictions_v5.csv

It does not place orders. Entry and exit orders remain controlled by
live/news_stream.py and live/exit_manager.py.
