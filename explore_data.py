"""
Fase 1 (lanjutan) - Advanced EDA: Statistik Profesional + Visualisasi Lengkap
V2.1 - Dengan Regime Detection, Parquet Export, & Sanity Check
Untuk persiapan Agentic AI Bot
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Setup style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

DATA_CSV = Path(__file__).parent / "data" / "btc_usdt_1d.csv"
DATA_PARQUET = Path(__file__).parent / "data" / "btc_usdt_1d.parquet"


# ==========================================
# 🍒 FUNGSI BARU V2.1
# ==========================================

def compute_hurst(ts, max_lag=100):
    """
    Hitung Hurst Exponent via variance of differences.
    Input: log price series (bukan return!)
    H > 0.5 = Trending (ikuti arah)
    H < 0.5 = Mean-reverting (balik ke rata-rata)
    H ≈ 0.5 = Random walk (acak)
    """
    lags = range(2, min(max_lag, len(ts) // 2))
    tau = []
    for lag in lags:
        diff = np.subtract(ts[lag:], ts[:-lag])
        tau.append(np.std(diff))

    tau = np.array(tau)
    lags_arr = np.array(list(lags))
    valid = (tau > 0) & np.isfinite(tau)

    if valid.sum() < 2:
        return 0.5  # Fallback: random walk

    m = np.polyfit(np.log(lags_arr[valid]), np.log(tau[valid]), 1)
    return m[0]


def sanity_check(df):
    """
    Validasi data 100% aman untuk dimakan backtesting engine / AI.
    Return: list of issues (kosong = semua aman).
    """
    issues = []

    # 1. Cek kolom kritis ada & tidak NaN
    critical_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in critical_cols:
        if col not in df.columns:
            issues.append(f"Kolom '{col}' TIDAK ADA di dataframe!")
        else:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                issues.append(f"Kolom '{col}' punya {nan_count} nilai NaN")

    # 2. Cek data terurut berdasarkan waktu
    if not df['date'].is_monotonic_increasing:
        issues.append("Data TIDAK terurut berdasarkan waktu!")

    # 3. Cek duplikat timestamp
    dup_count = df['date'].duplicated().sum()
    if dup_count > 0:
        issues.append(f"Ditemukan {dup_count} timestamp duplikat!")

    # 4. Cek candle corrupt (high < low = mustahil)
    bad_candles = (df['high'] < df['low']).sum()
    if bad_candles > 0:
        issues.append(f"Ditemukan {bad_candles} candle corrupt (high < low)!")

    # 5. Cek volume negatif
    neg_vol = (df['volume'] < 0).sum()
    if neg_vol > 0:
        issues.append(f"Ditemukan {neg_vol} volume negatif!")

    return issues


# ==========================================
# 1. LOAD & NORMALISASI DATA
# ==========================================
print("=" * 60)
print("MEMUAT DATA BTC...")
print("=" * 60)

df = pd.read_csv(DATA_CSV)

# Normalisasi schema (kompatibel dengan fetch_data.py)
if df.index.name == "datetime" or "datetime" in df.columns:
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["datetime"])
elif "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
elif "open_time_formatted" in df.columns:
    df["date"] = pd.to_datetime(df["open_time_formatted"])
else:
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

df = df.sort_values("date").reset_index(drop=True)
df["ret"] = df["close"].pct_change()
df = df.dropna(subset=["ret"])  # Hapus baris pertama yang NaN


# ==========================================
# 2. STATISTIK DASAR
# ==========================================
ann_vol = df["ret"].std() * np.sqrt(365)
cummax = df["close"].cummax()
max_dd = (df["close"] / cummax - 1).min()

print(f"\n📊 STATISTIK DASAR")
print(f"-" * 60)
print(f"Total baris       : {len(df)}")
print(f"Periode           : {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
print(f"Harga terakhir    : ${df['close'].iloc[-1]:,.0f}")
print(f"Total return      : {df['close'].iloc[-1] / df['close'].iloc[0] - 1:+.2%}")
print(f"Volatilitas ann.  : {ann_vol:.1%}")
print(f"Max drawdown      : {max_dd:.1%}")
print(f"Skewness return   : {df['ret'].skew():.2f}  (>0 = ekor kanan tebal)")


# ==========================================
# 3. METRIK PERFORMA PROFESIONAL
# ==========================================
print(f"\n📈 METRIK PERFORMA PROFESIONAL")
print(f"-" * 60)

total_days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
years = total_days / 365.25
total_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1
ann_return = (1 + total_return) ** (1 / years) - 1

sharpe_ratio = ann_return / ann_vol if ann_vol > 0 else 0

downside_ret = df["ret"][df["ret"] < 0]
downside_vol = downside_ret.std() * np.sqrt(365)
sortino_ratio = ann_return / downside_vol if downside_vol > 0 else 0

calmar_ratio = ann_return / abs(max_dd) if max_dd != 0 else 0

win_rate = (df["ret"] > 0).mean()

gross_profit = df["ret"][df["ret"] > 0].sum()
gross_loss = abs(df["ret"][df["ret"] < 0].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

def longest_streak(series, condition):
    streaks = []
    current_streak = 0
    for val in series:
        if condition(val):
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)
    return max(streaks) if streaks else 0

longest_win_streak = longest_streak(df["ret"], lambda x: x > 0)
longest_loss_streak = longest_streak(df["ret"], lambda x: x < 0)

kurtosis = df["ret"].kurtosis()

print(f"Return tahunan    : {ann_return:.2%}")
print(f"Sharpe Ratio      : {sharpe_ratio:.2f}  (>1 bagus, >2 luar biasa)")
print(f"Sortino Ratio     : {sortino_ratio:.2f}  (fokus ke downside risk)")
print(f"Calmar Ratio      : {calmar_ratio:.2f}  (return/max_dd)")
print(f"Win Rate          : {win_rate:.1%}")
print(f"Profit Factor     : {profit_factor:.2f}  (>1 profit, <1 rugi)")
print(f"Longest Win Streak : {longest_win_streak} hari")
print(f"Longest Loss Streak: {longest_loss_streak} hari")
print(f"Kurtosis          : {kurtosis:.2f}  (>3 = fat tails/ekor gemuk)")


# ==========================================
# 🍒 4. REGIME DETECTION (HURST EXPONENT) - BARU V2.1
# ==========================================
print(f"\n🧠 REGIME DETECTION (HURST EXPONENT)")
print(f"-" * 60)

# Hurst dihitung dari log price, bukan return!
log_prices = np.log(df["close"].values)
hurst = compute_hurst(log_prices, max_lag=100)

print(f"Hurst Exponent    : {hurst:.4f}")
if hurst > 0.55:
    regime = "TRENDING"
    regime_emoji = "📈"
    regime_advice = "Strategi Momentum/Breakout cocok. Ikuti arah tren."
elif hurst < 0.45:
    regime = "MEAN-REVERTING"
    regime_emoji = "🔄"
    regime_advice = "Strategi Mean Reversion cocok. Beli di support, jual di resistance."
else:
    regime = "RANDOM WALK"
    regime_emoji = "🎲"
    regime_advice = "Pasar acak. Sulit dieksploitasi dengan strategi sederhana."

print(f"Regime Pasar      : {regime_emoji} {regime}")
print(f"Saran Strategi    : {regime_advice}")


# ==========================================
# 5. VISUALISASI 1: ORIGINAL 3-PANEL
# ==========================================
print(f"\n📊 MEMBUAT VISUALISASI...")

fig1, axes1 = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.2, 1.2]})

axes1[0].plot(df["date"], df["close"], lw=1.2, color='#2E86AB')
axes1[0].set_ylabel("Harga (USDT)", fontweight='bold')
axes1[0].set_title("BTC/USD - Analisis Historis Lengkap (V2.1)",
                    fontsize=14, fontweight='bold', pad=20)
axes1[0].grid(alpha=0.3)

colors = np.where(df["ret"] >= 0, "#06A77D", "#D7263D")
axes1[1].bar(df["date"], df["ret"] * 100, color=colors, width=1.0, alpha=0.7)
axes1[1].set_ylabel("Return Harian (%)", fontweight='bold')
axes1[1].axhline(0, color="gray", lw=0.8)
axes1[1].grid(alpha=0.3)

drawdown = (df["close"] / cummax - 1) * 100
axes1[2].fill_between(df["date"], drawdown, 0, alpha=0.5, color='#D7263D')
axes1[2].set_ylabel("Drawdown (%)", fontweight='bold')
axes1[2].set_xlabel("Tanggal")
axes1[2].grid(alpha=0.3)

plt.tight_layout()
fig1.savefig(Path(__file__).parent / "data" / "exploration_basic.png",
             dpi=150, bbox_inches='tight')
print("✅ Saved: data/exploration_basic.png")


# ==========================================
# 6. VISUALISASI 2: ADVANCED METRICS (4-PANEL)
# ==========================================
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Rolling Volatility (30-day)
rolling_vol = df["ret"].rolling(30).std() * np.sqrt(365) * 100
axes2[0, 0].plot(df["date"], rolling_vol, color='#A23B72', lw=1.5)
axes2[0, 0].axhline(ann_vol * 100, color='gray', linestyle='--',
                     label=f'Avg: {ann_vol*100:.1f}%')
axes2[0, 0].set_title("Rolling Volatility (30-day)", fontweight='bold')
axes2[0, 0].set_ylabel("Volatilitas (%)")
axes2[0, 0].legend()
axes2[0, 0].grid(alpha=0.3)

# Panel 2: Return Distribution + Normal Curve
axes2[0, 1].hist(df["ret"] * 100, bins=50, density=True, alpha=0.6,
                  color='#F18F01', edgecolor='black')
xmin, xmax = axes2[0, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, df["ret"].mean() * 100, df["ret"].std() * 100)
axes2[0, 1].plot(x, p, 'k-', linewidth=2, label='Normal Distribution')
axes2[0, 1].set_title(f"Return Distribution (Kurtosis: {kurtosis:.2f})",
                       fontweight='bold')
axes2[0, 1].set_xlabel("Return Harian (%)")
axes2[0, 1].set_ylabel("Density")
axes2[0, 1].legend()
axes2[0, 1].grid(alpha=0.3)

# Panel 3: Autocorrelation (ACF) - 20 lags
from statsmodels.tsa.stattools import acf
acf_values = acf(df["ret"], nlags=20, fft=True)
axes2[1, 0].bar(range(21), acf_values, color='#C73E1D', alpha=0.7)
axes2[1, 0].axhline(0, color='black', lw=0.8)
confidence = 1.96 / np.sqrt(len(df))
axes2[1, 0].axhline(confidence, color='gray', linestyle='--',
                     label='95% Confidence')
axes2[1, 0].axhline(-confidence, color='gray', linestyle='--')
axes2[1, 0].set_title("Autocorrelation (ACF) - 20 Lags", fontweight='bold')
axes2[1, 0].set_xlabel("Lag (hari)")
axes2[1, 0].set_ylabel("Autocorrelation")
axes2[1, 0].legend()
axes2[1, 0].grid(alpha=0.3)

# Panel 4: Monthly Returns Heatmap
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
monthly_ret = df.groupby(['year', 'month'])['ret'].apply(
    lambda x: (1 + x).prod() - 1
)
monthly_ret = monthly_ret.unstack() * 100

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sns.heatmap(monthly_ret, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=axes2[1, 1], cbar_kws={'label': 'Return (%)'}, linewidths=0.5)
axes2[1, 1].set_title("Monthly Returns Heatmap (%)", fontweight='bold')
axes2[1, 1].set_xlabel("Bulan")
axes2[1, 1].set_ylabel("Tahun")
# Hanya set label jika jumlah kolom cocok
if len(monthly_ret.columns) == 12:
    axes2[1, 1].set_xticklabels(month_labels)

plt.tight_layout()
fig2.savefig(Path(__file__).parent / "data" / "exploration_advanced.png",
             dpi=150, bbox_inches='tight')
print("✅ Saved: data/exploration_advanced.png")


# ==========================================
# 7. INSIGHT UNTUK AGENTIC AI
# ==========================================
print(f"\n🤖 INSIGHT UNTUK AGENTIC AI BOT")
print(f"-" * 60)

print(f"1. MARKET REGIME (Hurst = {hurst:.4f}):")
print(f"   {regime_emoji} {regime} → {regime_advice}")

print(f"\n2. VOLATILITY:")
if ann_vol > 0.6:
    print(f"   ⚠️  Volatilitas TINGGI ({ann_vol:.1%}) → Position size kecil, stop-loss longgar")
else:
    print(f"   ✅ Volatilitas NORMAL ({ann_vol:.1%}) → Position size standar OK")

print(f"\n3. DISTRIBUTION:")
if kurtosis > 3:
    print(f"   ⚠️  Fat tails (kurtosis={kurtosis:.2f}) → Hindari model asumsi normal")
else:
    print(f"   ✅ Distribusi mendekati normal → Mean reversion bisa dicoba")

print(f"\n4. RISK MANAGEMENT:")
print(f"   Max Drawdown historis: {max_dd:.1%}")
print(f"   → Set toleransi drawdown bot minimal 2x lipat untuk safety margin")

print(f"\n5. SEASONALITY:")
best_month = monthly_ret.mean().idxmax()
worst_month = monthly_ret.mean().idxmin()
print(f"   Bulan terbaik  : {month_labels[best_month-1]} "
      f"(avg {monthly_ret.mean()[best_month]:.1f}%)")
print(f"   Bulan terburuk : {month_labels[worst_month-1]} "
      f"(avg {monthly_ret.mean()[worst_month]:.1f}%)")


# ==========================================
# 🍒 8. EXPORT KE PARQUET - BARU V2.1
# ==========================================
print(f"\n💾 EXPORT DATA...")
print(f"-" * 60)

try:
    # Bersihkan kolom helper sebelum export
    df_export = df.drop(columns=['year', 'month'], errors='ignore')
    df_export.to_parquet(DATA_PARQUET, index=False, engine='pyarrow')
    size_csv = DATA_CSV.stat().st_size / 1024
    size_pq = DATA_PARQUET.stat().st_size / 1024
    compression = (1 - size_pq / size_csv) * 100
    print(f"✅ Parquet saved: data/btc_usdt_1d.parquet")
    print(f"   CSV: {size_csv:.0f} KB → Parquet: {size_pq:.0f} KB "
          f"(hemat {compression:.0f}%)")
except ImportError:
    print("⚠️  Library 'pyarrow' belum terinstall. Skip export parquet.")
    print("   Install dengan: pip install pyarrow")
except Exception as e:
    print(f"⚠️  Gagal export parquet: {e}")


# ==========================================
# 🍒 9. DATA SANITY CHECK - BARU V2.1
# ==========================================
print(f"\n🔍 DATA SANITY CHECK")
print(f"-" * 60)

issues = sanity_check(df)

if len(issues) == 0:
    print("✅ [SANITY CHECK PASSED] Data 100% bersih dan siap untuk "
          "Feature Engineering & Backtesting!")
else:
    print(f"❌ [SANITY CHECK FAILED] Ditemukan {len(issues)} masalah:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    print("   ⚠️  Perbaiki masalah di atas sebelum lanjut ke tahap berikutnya!")


# ==========================================
# SELESAI
# ==========================================
print(f"\n{'=' * 60}")
print(f"✅ ANALISIS LENGKAP V2.1 SELESAI!")
print(f"{'=' * 60}")
print(f"\n📁 File yang dihasilkan:")
print(f"   📊 data/exploration_basic.png    (3-panel: harga, return, drawdown)")
print(f"   📊 data/exploration_advanced.png  (4-panel: vol, distribusi, ACF, heatmap)")
try:
    if DATA_PARQUET.exists():
        print(f"   💾 data/btc_usdt_1d.parquet       (data bersih, siap untuk AI)")
except:
    pass
print(f"\n🚀 Langkah selanjutnya: Feature Engineering (Indikator Teknikal)")
print(f"{'=' * 60}\n")