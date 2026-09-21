"""
Fase 2 - Strategi Puncak: Regime-Aware Backtest
Logika Adaptif:
  1. BULLISH_TREND   : Eksekusi SMA Crossover (Fast=20, Slow=50)
  2. SIDEWAYS_RANGING: Eksekusi RSI Mean Reversion (Buy > 30, Sell < 70)
  3. BEARISH_TREND   : Stay in Cash (Force Sell jika sedang hold)

Aturan Eksekusi:
  - Sinyal di-close hari t -> Eksekusi di OPEN hari t+1 (Anti look-ahead bias)
  - Fee 0.1% + Slippage 0.05% per transaksi
  - Long-only, all-in/all-out
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

DATA = Path(__file__).parent / "data" / "btc_processed_1d.csv"
FEE = 0.001
SLIPPAGE = 0.0005
COST = FEE + SLIPPAGE


def load_data() -> pd.DataFrame:
    """Load data yang sudah diproses oleh feature_pipeline.py"""
    df = pd.read_csv(DATA)
    
    # Normalisasi kolom tanggal
    if df.index.name == "datetime" or "datetime" in df.columns:
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        
    return df.sort_values("date").reset_index(drop=True)


def backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    df = df.copy()
    
    # Pastikan kolom indikator ada (fallback jika belum di-run feature_pipeline)
    if 'sma_20' not in df.columns:
        df['sma_20'] = df['close'].rolling(20).mean()
    if 'sma_50' not in df.columns:
        df['sma_50'] = df['close'].rolling(50).mean()
    if 'rsi_14' not in df.columns:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
        df['rsi_14'] = 100 - (100 / (1 + gain / loss))

    cash, btc, in_pos = 1.0, 0.0, False
    trades, equity = [], []
    
    # Mulai loop dari hari ke-50 (agar SMA 50 sudah valid, tidak NaN)
    start_idx = 50 
    
    for i in range(start_idx, len(df)):
        o = df.at[i, "open"]
        date = df.at[i, "date"]
        regime = df.at[i, "regime"]
        
        # Ambil nilai indikator hari ini (i) dan kemarin (i-1) untuk deteksi crossover
        sma20 = df.at[i, "sma_20"]
        sma50 = df.at[i, "sma_50"]
        sma20_prev = df.at[i-1, "sma_20"]
        sma50_prev = df.at[i-1, "sma_50"]
        
        rsi = df.at[i, "rsi_14"]
        rsi_prev = df.at[i-1, "rsi_14"]
        
        signal = "HOLD"
        
        # ==========================================
        # LOGIKA REGIME-AWARE
        # ==========================================
        if not in_pos:
            # Cari posisi BELI
            if regime == 'BULLISH_TREND':
                # SMA Crossover: Fast cross ke ATAS Slow
                if sma20 > sma50 and sma20_prev <= sma50_prev:
                    signal = "BUY"
            elif regime == 'SIDEWAYS_RANGING':
                # RSI Mean Reversion: RSI cross ke ATAS 30 (keluar dari oversold)
                if rsi > 30 and rsi_prev <= 30:
                    signal = "BUY"
        else:
            # Cari posisi JUAL
            if regime == 'BEARISH_TREND':
                # PANIC SELL: Keluar segera saat market jadi bearish
                signal = "SELL"
            elif regime == 'BULLISH_TREND':
                # SMA Crossover: Fast cross ke BAWAH Slow
                if sma20 < sma50 and sma20_prev >= sma50_prev:
                    signal = "SELL"
            elif regime == 'SIDEWAYS_RANGING':
                # RSI Mean Reversion: RSI cross ke BAWAH 70 (keluar dari overbought)
                if rsi < 70 and rsi_prev >= 70:
                    signal = "SELL"
                    
        # ==========================================
        # EKSEKUSI TRADE (di harga OPEN hari ini)
        # ==========================================
        if signal == "BUY" and not in_pos:
            btc = cash * (1 - COST) / o
            cash = 0.0
            in_pos = True
            trades.append(("BUY", date, o, regime))
            
        elif signal == "SELL" and in_pos:
            cash = btc * o * (1 - COST)
            btc = 0.0
            in_pos = False
            trades.append(("SELL", date, o, regime))
            
        # Hitung equity harian
        equity.append(cash + btc * o)
        
    # Handle jika di akhir periode masih dalam posisi
    if in_pos:
        cash = btc * df["close"].iloc[-1] * (1 - COST)
        trades.append(("FORCE_SELL_END", df["date"].iloc[-1], df["close"].iloc[-1], "END"))
        
    df = df.iloc[start_idx:].copy()
    df["equity"] = equity
    
    # ==========================================
    # HITUNG METRIK
    # ==========================================
    ret = df["equity"].pct_change().dropna()
    bh = df["close"] / df["close"].iloc[0]
    n_years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    dd = df["equity"] / df["equity"].cummax() - 1
    
    closed_trades = [t for t in trades if t[0] in ("SELL", "FORCE_SELL_END")]
    opened_trades = [t for t in trades if t[0] == "BUY"]
    
    # Hitung win rate & avg profit/loss (dalam PERSEN, fix bug sebelumnya)
    wins = 0
    profits_pct = []
    for s, b in zip(closed_trades, opened_trades):
        if s[0] == "FORCE_SELL_END":
            continue
        trade_ret = (s[2] - b[2]) / b[2]
        profits_pct.append(trade_ret)
        if trade_ret > 0:
            wins += 1
            
    win_rate = wins / len(profits_pct) if profits_pct else 0
    avg_profit = np.mean([p for p in profits_pct if p > 0]) if any(p > 0 for p in profits_pct) else 0
    avg_loss = np.mean([p for p in profits_pct if p < 0]) if any(p < 0 for p in profits_pct) else 0

    metrics = {
        "total_return": df["equity"].iloc[-1] - 1,
        "cagr": (df["equity"].iloc[-1]) ** (1 / n_years) - 1,
        "ann_vol": ret.std() * np.sqrt(365),
        "sharpe": (ret.mean() / ret.std() * np.sqrt(365)) if ret.std() > 0 else 0,
        "max_dd": dd.min(),
        "n_trades": len(opened_trades),
        "bh_return": bh.iloc[-1] - 1,
        "bh_max_dd": (bh / bh.cummax() - 1).min(),
        "win_rate": win_rate,
        "avg_profit_pct": avg_profit,
        "avg_loss_pct": avg_loss,
    }
    
    return df, metrics, trades


def report(m: dict) -> None:
    print("=" * 75)
    print("🧠 HASIL BACKTEST: REGIME-AWARE STRATEGY (SMA + RSI + CASH)")
    print("=" * 75)
    print(f"TOTAL RETURN    : {m['total_return']:+.2%}   (Buy & Hold: {m['bh_return']:+.2%})")
    print(f"CAGR            : {m['cagr']:+.2%}")
    print(f"SHARPE RATIO    : {m['sharpe']:.2f}   (Target: > 1.0)")
    print(f"MAX DRAWDOWN    : {m['max_dd']:.1%}   (Buy & Hold: {m['bh_max_dd']:.1%})")
    print(f"VOLATILITAS ANN : {m['ann_vol']:.1%}")
    print("-" * 75)
    print(f"JUMLAH TRADE    : {m['n_trades']}")
    print(f"WIN RATE        : {m['win_rate']:.1%}")
    print(f"AVG PROFIT/TRADE: {m['avg_profit_pct']:+.2%}")
    print(f"AVG LOSS/TRADE  : {m['avg_loss_pct']:.2%}")
    print("=" * 75)


def plot_results(df: pd.DataFrame, trades: list) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [1.5, 1]})

    # --- PANEL 1: EQUITY CURVE ---
    bh = df["close"] / df["close"].iloc[0]
    axes[0].plot(df["date"], df["equity"], label="Regime-Aware Bot",
                 lw=1.5, color='#2E86AB')
    axes[0].plot(df["date"], bh, label="Buy & Hold",
                 lw=1.2, alpha=0.6, color='#F18F01')
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1 (Log Scale)", fontweight='bold')
    axes[0].set_title("Regime-Aware Strategy vs Buy & Hold", fontweight='bold', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- PANEL 2: HARGA + REGIME BACKGROUND + TRADE MARKERS ---
    axes[1].plot(df["date"], df["close"], color='black', lw=1, alpha=0.5)
    
    # Warnai background berdasarkan regime
    colors = {'BULLISH_TREND': '#d4edda', 'SIDEWAYS_RANGING': '#fff3cd', 'BEARISH_TREND': '#f8d7da'}
    labels = {'BULLISH_TREND': 'Bullish', 'SIDEWAYS_RANGING': 'Sideways', 'BEARISH_TREND': 'Bearish'}
    
    # Buat custom legend untuk regime
    patches = [mpatches.Patch(color=colors[r], label=labels[r]) for r in colors]
    axes[1].legend(handles=patches, loc='upper left', fontsize=9)

    # Fill between untuk visualisasi regime
    for i in range(1, len(df)):
        date_prev = df["date"].iloc[i-1]
        date_curr = df["date"].iloc[i]
        regime = df["regime"].iloc[i]
        axes[1].axvspan(date_prev, date_curr, color=colors[regime], alpha=0.4)

    # Plot marker BUY/SELL
    for trade in trades:
        action, date, price, regime = trade
        if action == "BUY":
            axes[1].scatter(date, price, marker='^', color='green', s=60, zorder=5, edgecolor='black')
        elif action in ("SELL", "FORCE_SELL_END"):
            axes[1].scatter(date, price, marker='v', color='red', s=60, zorder=5, edgecolor='black')

    axes[1].set_ylabel("BTC Price (USDT)", fontweight='bold')
    axes[1].set_xlabel("Date", fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / "data" / "backtest_regime_aware.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved -> {out}")


if __name__ == "__main__":
    print("🚀 Memulai Regime-Aware Backtest...\n")
    df = load_data()
    df_processed, metrics, trades = backtest(df)
    report(metrics)
    plot_results(df_processed, trades)
    print("\n✅ Selesai! Silakan cek data/backtest_regime_aware.png")