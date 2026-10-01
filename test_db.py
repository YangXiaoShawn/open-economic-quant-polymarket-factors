import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traceback
import db

tests = [
    ("get_events",              lambda: db.get_events()),
    ("get_markets(2 cols)",     lambda: db.get_markets("market_id, question")),
    ("get_user_features(3col)", lambda: db.get_user_features("user_address, n_trades, frac_maker")),
    ("get_trades 1 month",     lambda: db.get_trades("2024-11-01", "2024-12-01", "trade_id,timestamp,price,quantity")),
    ("get_pnl_daily 1 month",  lambda: db.get_pnl_daily("2024-11-01", "2024-12-01")),
    ("get_ctf resolutions",    lambda: db.get_ctf("resolutions")),
    ("get_order_filled 1mo",   lambda: db.get_order_filled("2024_11", "2024_11")),
    ("get_ohlcv_1d(filter)",   lambda: db.get_ohlcv_1d([100, 200, 300])),
]

for name, fn in tests:
    try:
        df = fn()
        print(f"OK   {name}: {len(df):,} rows x {len(df.columns)} cols")
        del df
    except Exception as e:
        print(f"FAIL {name}: {e}")
        traceback.print_exc()

print("Done.")
