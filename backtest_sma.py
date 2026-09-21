"""Fase 2 - Strategi pertama: SMA crossover backtest (realistis).

Aturan:
  - Long-only: beli BTC saat SMA fast cross ke ATAS SMA slow, jual saat cross ke BAWAH
  - Eksekusi di OPEN hari berikutnya (sinyal close hari ini -> eksekusi besok)
  - Biaya: fee 0.1% + slippage 0.05% per transaksi
  - Tidak pakai leverage, all-in/all-out (position sizing menyusul di fase 4)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "btc_usdt_1d.csv"
FEE = 0.001        # 0.1% per transaksi (taker Binance)
SLIPPAGE = 0.0005  # 0.05% slippage per transaksi
FAST, SLOW = 20, 50

def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    # Kompatibel dengan output fetch_data.py (kolom 'datetime' sebagai index)
    if df.index.name == "datetime" or "datetime" in df.columns:
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "open_time_formatted" in df.columns:
        df["date"] = pd.to_datetime(df["open_time_formatted"])
    else:
        df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.sort_values("date").reset_index(drop=True)

def backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(FAST).mean()
    df["sma_slow"] = df["close"].rolling(SLOW).mean()

    # sinyal di close hari t -> eksekusi di open hari t+1
    cross_up = (df["sma_fast"] > df["sma_slow"]) & (df["sma_fast"].shift(1) <= df["sma_slow"].shift(1))
    cross_dn = (df["sma_fast"] < df["sma_slow"]) & (df["sma_fast"].shift(1) >= df["sma_slow"].shift(1))

    cash, btc, in_pos = 1.0, 0.0, False
    trades, equity = [], []
    cost = FEE + SLIPPAGE

    for i in range(len(df)):
        o = df.at[i, "open"]
        if i > 0 and cross_up.iloc[i - 1] and not in_pos:
            btc = cash * (1 - cost) / o
            cash, in_pos = 0.0, True
            trades.append(("BUY", df.at[i, "date"], o))
        elif i > 0 and cross_dn.iloc[i - 1] and in_pos:
            cash = btc * o * (1 - cost)
            btc, in_pos = 0.0, False
            trades.append(("SELL", df.at[i, "date"], o))
        equity.append(cash + btc * o)

    df["equity"] = equity
    if in_pos:  # tutup posisi terakhir di close terakhir buat itung metrik
        df.loc[df.index[-1], "equity"] = btc * df["close"].iloc[-1]

    ret = df["equity"].pct_change().dropna()
    bh = df["close"] / df["close"].iloc[0]
    n_years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365
    dd = df["equity"] / df["equity"].cummax() - 1
    closed = [t for t in trades if t[0] == "SELL"]
    opened = [t for t in trades if t[0] == "BUY"]
    wins = sum(1 for s, b in zip(closed, opened) if s[2] > b[2])

    metrics = {
        "total_return": df["equity"].iloc[-1] - 1,
        "cagr": (df["equity"].iloc[-1]) ** (1 / n_years) - 1,
        "ann_vol": ret.std() * np.sqrt(365),
        "sharpe": ret.mean() / ret.std() * np.sqrt(365),
        "max_dd": dd.min(),
        "n_trades": len(trades),
        "exposure": (df["sma_fast"] > df["sma_slow"]).mean(),
        "bh_return": bh.iloc[-1] - 1,
        "bh_max_dd": (bh / bh.cummax() - 1).min(),
        "win_rate": wins / len(closed) if closed else float("nan"),
    }
    return df, metrics


def report(df: pd.DataFrame, m: dict) -> None:
    print("=== Hasil backtest SMA crossover (fee + slippage included) ===")
    print(f"total return : {m['total_return']:+.2%}   (buy & hold: {m['bh_return']:+.2%})")
    print(f"CAGR         : {m['cagr']:+.2%}")
    print(f"vol ann.     : {m['ann_vol']:.1%}")
    print(f"sharpe       : {m['sharpe']:.2f}")
    print(f"max drawdown : {m['max_dd']:.1%}   (buy & hold: {m['bh_max_dd']:.1%})")
    print(f"jumlah trade : {m['n_trades']}")
    print(f"exposure     : {m['exposure']:.0%} of the time (uang kerja, sisanya idle di cash)")
    print(f"win rate     : {m['win_rate']:.0%}")


if __name__ == "__main__":
    df, m = backtest(load())
    report(df, m)
    bh = df["close"] / df["close"].iloc[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["equity"], label="SMA crossover", lw=1.4)
    ax.plot(df["date"], bh, label="Buy & hold", lw=1.2, alpha=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = Path(__file__).parent / "data" / "backtest_equity.png"
    plt.savefig(out, dpi=110)
    print(f"plot saved -> {out}")