"""
feature_pipeline.py - Pipeline Feature Engineering + Market Regime Detection
V2.0 - Integrasi dengan indicators.py untuk Agentic AI

Fungsi:
1. Hitung SEMUA indikator teknikal (via indicators.py)
2. Deteksi Market Regime (Bullish/Bearish/Sideways)
3. Multi-timeframe processing (1D + 1H)
4. Export data siap untuk AI/Backtesting

"""
from pathlib import Path
import numpy as np
import pandas as pd

# Import library indikator kita
from indicators import ema, atr, rsi, historical_volatility, add_all_indicators


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung fitur teknikal menggunakan indicators.py (lebih akurat).
    Plus tambahan fitur unik dari file lama: ATR percentage.
    """
    df = df.copy()
    
    # Pakai fungsi dari indicators.py (lebih akurat)
    df['ema_50'] = ema(df['close'], 50)
    df['ema_200'] = ema(df['close'], 200)
    df['rsi_14'] = rsi(df['close'], 14)
    df['atr_14'] = atr(df['high'], df['low'], df['close'], 14)
    df['hist_vol_30'] = historical_volatility(df['close'], 30)
    
    # Fitur unik dari file lama: ATR Percentage (Relative Volatility)
    df['atr_pct'] = (df['atr_14'] / df['close']) * 100
    
    return df


def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Klasifikasi kondisi pasar BTC berdasarkan struktur harga.
    
    Logika:
    - BULLISH_TREND: Harga > EMA50 > EMA200 (uptrend kuat)
    - BEARISH_TREND: Harga < EMA50 < EMA200 (downtrend kuat)
    - SIDEWAYS_RANGING: Tidak memenuhi kriteria tren (konsolidasi)
    
    Ini KRITIS untuk Agentic AI - AI bisa switch strategi berdasarkan regime!
    """
    df = calculate_technical_features(df)
    
    conditions = [
        # Bullish: Harga di atas EMA50, EMA50 di atas EMA200
        (df['close'] > df['ema_50']) & (df['ema_50'] > df['ema_200']),
        
        # Bearish: Harga di bawah EMA50, EMA50 di bawah EMA200
        (df['close'] < df['ema_50']) & (df['ema_50'] < df['ema_200']),
    ]
    
    choices = ['BULLISH_TREND', 'BEARISH_TREND']
    
    # Default: Sideways
    df['regime'] = np.select(conditions, choices, default='SIDEWAYS_RANGING')
    
    return df


def process_multi_timeframe(df_1d: pd.DataFrame, 
                             df_1h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Proses data 1D dan 1H sekaligus.
    
    Konsep:
    - 1D = Big picture (regime utama)
    - 1H = Timing entry/exit (presisi)
    
    Nanti AI bisa pakai: "Kalau 1D bullish, cari entry di 1H saat pullback."
    """
    processed_1d = detect_market_regime(df_1d)
    processed_1h = detect_market_regime(df_1h)
    
    return processed_1d, processed_1h


def add_all_features_for_ai(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan SEMUA fitur (indikator + regime) siap untuk AI.
    Ini "one-stop function" untuk prepare data sebelum masuk ke model AI.
    """
    # Tambah semua indikator dari indicators.py
    df = add_all_indicators(df)
    
    # Tambah regime detection
    df = detect_market_regime(df)
    
    # Encode regime jadi numeric (untuk AI)
    regime_map = {
        'BULLISH_TREND': 1,
        'SIDEWAYS_RANGING': 0,
        'BEARISH_TREND': -1
    }
    df['regime_numeric'] = df['regime'].map(regime_map)
    
    return df


def print_regime_analysis(df: pd.DataFrame, timeframe: str = "1D") -> None:
    """Print analisis regime untuk quick inspection."""
    print(f"\n{'='*60}")
    print(f"📊 MARKET REGIME ANALYSIS ({timeframe})")
    print(f"{'='*60}")
    
    # Distribusi regime
    print(f"\nDistribusi Regime:")
    regime_counts = df['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        print(f"  {regime:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Latest state
    latest = df.iloc[-1]
    print(f"\nLatest Market State:")
    print(f"  Date       : {df.index[-1] if isinstance(df.index[-1], str) else df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Close      : ${latest['close']:,.2f}")
    print(f"  Regime     : {latest['regime']}")
    print(f"  RSI (14)   : {latest['rsi_14']:.2f}")
    print(f"  ATR (%)    : {latest['atr_pct']:.2f}%")
    print(f"  EMA 50     : ${latest['ema_50']:,.2f}")
    print(f"  EMA 200    : ${latest['ema_200']:,.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("🚀 Processing historical data for Pattern Engine V2.0...")
    
    data_dir = Path(__file__).parent / "data"
    
    try:
        # Load data 1D dan 1H
        df_1d = pd.read_csv(data_dir / 'btc_usdt_1d.csv', 
                            index_col='datetime' if 'datetime' in pd.read_csv(data_dir / 'btc_usdt_1d.csv', nrows=0).columns else 0,
                            parse_dates=True)
        df_1h = pd.read_csv(data_dir / 'btc_usdt_1h.csv',
                            index_col='datetime' if 'datetime' in pd.read_csv(data_dir / 'btc_usdt_1h.csv', nrows=0).columns else 0,
                            parse_dates=True)
        
        # Reset index kalau perlu (kompatibilitas)
        if df_1d.index.name != 'datetime':
            df_1d = df_1d.reset_index()
            df_1d['datetime'] = pd.to_datetime(df_1d.iloc[:, 0])
            df_1d = df_1d.set_index('datetime')
        
        if df_1h.index.name != 'datetime':
            df_1h = df_1h.reset_index()
            df_1h['datetime'] = pd.to_datetime(df_1h.iloc[:, 0])
            df_1h = df_1h.set_index('datetime')
        
        # Proses multi-timeframe
        processed_1d, processed_1h = process_multi_timeframe(df_1d, df_1h)
        
        # Print analisis
        print_regime_analysis(processed_1d, "1D")
        print_regime_analysis(processed_1h, "1H")
        
        # Simpan data yang sudah di-feature-engineering
        processed_1d.to_csv(data_dir / 'btc_processed_1d.csv')
        processed_1h.to_csv(data_dir / 'btc_processed_1h.csv')
        
        print(f"✅ Processed datasets saved:")
        print(f"   📁 data/btc_processed_1d.csv ({len(processed_1d)} rows)")
        print(f"   📁 data/btc_processed_1h.csv ({len(processed_1h)} rows)")
        print(f"\n🎯 Next step: Backtest dengan regime-aware strategy!")
        
    except FileNotFoundError as e:
        print(f"❌ Data CSV tidak ditemukan: {e}")
        print("   Pastikan sudah menjalankan fetch_data.py!")