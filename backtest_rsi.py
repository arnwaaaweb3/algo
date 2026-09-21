"""
Fase 2 - Strategi kedua: RSI Mean Reversion
Aturan:
  - Long-only
  - BELI saat RSI < 30 (oversold) dan mulai naik (RSI cross ke atas 30)
  - JUAL saat RSI > 70 (overbought) dan mulai turun (RSI cross ke bawah 70)
  - Eksekusi di OPEN hari berikutnya (anti look-ahead bias)
  - Fee + slippage included
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import library indikator kita
from indicators import rsi

DATA = Path(__file__).parent / "data" / "btc_usdt_1d.csv"
FEE = 0.001
SLIPPAGE = 0.0005
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    if df.index.name == "datetime" or "datetime" in df.columns:
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    df = df.copy()
    df["rsi"] = rsi(df["close"], RSI_PERIOD)

    # Sinyal: RSI cross ke atas 30 (beli) atau cross ke bawah 70 (jual)
    cross_buy = (df["rsi"] > RSI_OVERSOLD) & (df["rsi"].shift(1) <= RSI_OVERSOLD)
    cross_sell = (df["rsi"] < RSI_OVERBOUGHT) & (df["rsi"].shift(1) >= RSI_OVERBOUGHT)

    cash, btc, in_pos = 1.0, 0.0, False
    trades, equity = [], []
    cost = FEE + SLIPPAGE

    for i in range(len(df)):
        o = df.at[i, "open"]
        if i > 0 and cross_buy.iloc[i - 1] and not in_pos:
            btc = cash * (1 - cost) / o
            cash, in_pos = 0.0, True
            trades.append(("BUY", df.at[i, "date"], o, df.at[i-1, "rsi"]))
        elif i > 0 and cross_sell.iloc[i - 1] and in_pos:
            cash = btc * o * (1 - cost)
            btc, in_pos = 0.0, False
            trades.append(("SELL", df.at[i, "date"], o, df.at[i-1, "rsi"]))
        equity.append(cash + btc * o)

    df["equity"] = equity
    if in_pos:
        df.loc[df.index[-1], "equity"] = btc * df["close"].iloc[-1]

    ret = df["equity"].pct_change().dropna()
    bh = df["close"] / df["close"].iloc[0]
    n_years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365
    dd = df["equity"] / df["equity"].cummax() - 1

    closed = [t for t in trades if t[0] == "SELL"]
    opened = [t for t in trades if t[0] == "BUY"]
    wins = sum(1 for s, b in zip(closed, opened) if s[2] > b[2])

    # Hitung average profit/loss per trade (dalam PERSEN) - FIXED!
    profits_pct = [(s[2] - b[2]) / b[2] for s, b in zip(closed, opened)]
    avg_profit = np.mean([p for p in profits_pct if p > 0]) if any(p > 0 for p in profits_pct) else 0
    avg_loss = np.mean([p for p in profits_pct if p < 0]) if any(p < 0 for p in profits_pct) else 0

    metrics = {
        "total_return": df["equity"].iloc[-1] - 1,
        "cagr": (df["equity"].iloc[-1]) ** (1 / n_years) - 1,
        "ann_vol": ret.std() * np.sqrt(365),
        "sharpe": ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else 0,
        "max_dd": dd.min(),
        "n_trades": len(trades),
        "exposure": (df["rsi"] > RSI_OVERSOLD).mean() if in_pos else 0,
        "bh_return": bh.iloc[-1] - 1,
        "bh_max_dd": (bh / bh.cummax() - 1).min(),
        "win_rate": wins / len(closed) if closed else float("nan"),
        "avg_profit_pct": avg_profit,
        "avg_loss_pct": avg_loss,
    }
    return df, metrics, trades


def report(df: pd.DataFrame, m: dict) -> None:
    print("=" * 70)
    print("=== Hasil backtest RSI Mean Reversion (fee + slippage included) ===")
    print("=" * 70)
    print(f"Parameter       : RSI({RSI_PERIOD}), Oversold={RSI_OVERSOLD}, Overbought={RSI_OVERBOUGHT}")
    print(f"-" * 70)
    print(f"total return    : {m['total_return']:+.2%}   (buy & hold: {m['bh_return']:+.2%})")
    print(f"CAGR            : {m['cagr']:+.2%}")
    print(f"vol ann.        : {m['ann_vol']:.1%}")
    print(f"sharpe          : {m['sharpe']:.2f}")
    print(f"max drawdown    : {m['max_dd']:.1%}   (buy & hold: {m['bh_max_dd']:.1%})")
    print(f"-" * 70)
    print(f"jumlah trade    : {m['n_trades']}")
    print(f"win rate        : {m['win_rate']:.0%}")
    print(f"avg profit/trade: {m['avg_profit_pct']:+.2%}")
    print(f"avg loss/trade  : {m['avg_loss_pct']:.2%}")
    print("=" * 70)


def plot_results(df: pd.DataFrame, trades: list) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1, 1]})

    # Panel 1: Equity curve vs Buy & Hold
    bh = df["close"] / df["close"].iloc[0]
    axes[0].plot(df["date"], df["equity"], label="RSI Strategy",
                 lw=1.4, color='#2E86AB')
    axes[0].plot(df["date"], bh, label="Buy & Hold",
                 lw=1.2, alpha=0.7, color='#F18F01')
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1 (log)")
    axes[0].set_title("RSI Mean Reversion Strategy", fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: RSI dengan zona overbought/oversold
    axes[1].plot(df["date"], df["rsi"], color='#A23B72', lw=1)
    axes[1].axhline(RSI_OVERBOUGHT, color='red', linestyle='--',
                     alpha=0.7, label=f'Overbought ({RSI_OVERBOUGHT})')
    axes[1].axhline(RSI_OVERSOLD, color='green', linestyle='--',
                     alpha=0.7, label=f'Oversold ({RSI_OVERSOLD})')
    axes[1].fill_between(df["date"], RSI_OVERBOUGHT, 100,
                          alpha=0.1, color='red')
    axes[1].fill_between(df["date"], 0, RSI_OVERSOLD,
                          alpha=0.1, color='green')
    axes[1].set_ylabel("RSI")
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)

    # Panel 3: Harga dengan marker BUY/SELL
    axes[2].plot(df["date"], df["close"], color='gray', lw=1, alpha=0.7)
    for trade in trades:
        action, date, price, _ = trade
        if action == "BUY":
            axes[2].scatter(date, price, marker='^', color='green',
                            s=80, zorder=5)
        else:
            axes[2].scatter(date, price, marker='v', color='red',
                            s=80, zorder=5)
    axes[2].set_ylabel("BTC Price (USDT)")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / "data" / "backtest_rsi.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved -> {out}")


if __name__ == "__main__":
    df, m, trades = backtest(load())
    report(df, m)
    plot_results(df, trades)