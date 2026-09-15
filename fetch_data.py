"""Fase 1 - Data pipeline: ambil klines BTC/USDT dari Binance via ccxt.

Usage:
    python fetch_data.py --symbol BTC/USDT --timeframe 1d --days 730
"""
import argparse
import time
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_all(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Ambil klines historis dengan pagination (max 1000 bar per request)."""
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    all_rows = []

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        since = batch[-1][0] + 1  # lanjut setelah candle terakhir
        print(f"  fetched {len(all_rows)} candles...")
        if len(batch) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(
        all_rows,
        columns=["open_time", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Data pipeline tanpa validasi = bom waktu. Cek sebelum lanjut."""
    assert df[["open", "high", "low", "close", "volume"]].notna().all().all(), "ada NaN!"
    assert not df["date"].duplicated().any(), "ada tanggal duplikat!"
    assert (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all(), "high tidak valid!"
    assert (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all(), "low tidak valid!"
    print("  validasi OK: tidak ada NaN, duplikat, atau OHLC aneh.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--days", type=int, default=730)
    args = ap.parse_args()

    print(f"Fetching {args.symbol} {args.timeframe} ({args.days} hari)...")
    df = fetch_all(args.symbol, args.timeframe, args.days)
    validate(df)

    out = DATA_DIR / f"{args.symbol.replace('/', '_')}_{args.timeframe}.csv"
    df.to_csv(out, index=False)
    print(f"  saved -> {out} ({len(df)} baris)")
