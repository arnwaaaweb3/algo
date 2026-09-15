"""Fase 1 (lanjutan) - Eksplorasi data: statistik + visualisasi BTC."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "btc_usdt_1d.csv"

df = pd.read_csv(DATA)

# --- normalisasi schema ---
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
elif "open_time_formatted" in df.columns:
    df["date"] = pd.to_datetime(df["open_time_formatted"])
else:
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

df = df.sort_values("date").reset_index(drop=True)
df["ret"] = df["close"].pct_change()

# --- statistik dasar ---
ann_vol = df["ret"].std() * np.sqrt(365)
cummax = df["close"].cummax()
max_dd = (df["close"] / cummax - 1).min()

print(f"baris           : {len(df)}")
print(f"periode         : {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
print(f"harga terakhir  : ${df['close'].iloc[-1]:,.0f}")
print(f"total return    : {df['close'].iloc[-1] / df['close'].iloc[0] - 1:+.2%}")
print(f"volatilitas ann.: {ann_vol:.1%}")
print(f"max drawdown    : {max_dd:.1%}")
print(f"skewness return : {df['ret'].skew():.2f}  (>0 = ekor kanan tebal)")

# --- plot ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
axes[0].plot(df["date"], df["close"], lw=1.2)
axes[0].set_ylabel("Harga (USDT)")
axes[1].bar(df["date"], df["ret"] * 100,
            color=np.where(df["ret"] >= 0, "green", "red"), width=1.0)
axes[1].set_ylabel("Return harian (%)")
axes[1].axhline(0, color="gray", lw=0.8)
axes[2].fill_between(df["date"], (df["close"] / cummax - 1) * 100, 0, alpha=0.5)
axes[2].set_ylabel("Drawdown (%)")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "data" / "exploration.png", dpi=110)
print("plot saved -> data/exploration.png")