"""
indicators.py - Library Indikator Teknikal untuk Algo Trading Bot
V1.0 - Pure pandas (no TA-Lib dependency)

Semua fungsi:
- Return pandas Series atau DataFrame
- Handle NaN otomatis (warmup period)
- Vectorized (cepat)
- Siap untuk Agentic AI (bisa di-normalize)

"""
import numpy as np
import pandas as pd


# ==========================================
# 📊 MOVING AVERAGES
# ==========================================

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (lebih responsif dari SMA)"""
    return series.ewm(span=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average (bobot linier)"""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


# ==========================================
# 📈 MOMENTUM OSCILLATORS
# ==========================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (0-100)
    > 70 = overbought (potensi turun)
    < 30 = oversold (potensi naik)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence)
    Return DataFrame dengan kolom: macd, signal, histogram
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }, index=series.index)


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator (0-100)
    %K = fast line, %D = slow line (SMA of %K)
    > 80 = overbought, < 20 = oversold
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame({'stoch_k': k, 'stoch_d': d}, index=close.index)


# ==========================================
# 📉 VOLATILITY INDICATORS
# ==========================================

def bollinger_bands(series: pd.Series, period: int = 20,
                    std_dev: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands
    Return DataFrame: middle, upper, lower, bandwidth, %b
    %b = posisi harga relatif terhadap bands (0-1)
    """
    middle = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std()

    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)
    bandwidth = (upper - lower) / middle
    percent_b = (series - lower) / (upper - lower)

    return pd.DataFrame({
        'bb_middle': middle,
        'bb_upper': upper,
        'bb_lower': lower,
        'bb_bandwidth': bandwidth,
        'bb_percent_b': percent_b
    }, index=series.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """
    Average True Range - mengukur volatilitas
    Berguna untuk: stop-loss, position sizing, filter volatility
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.ewm(span=period, adjust=False).mean()


def historical_volatility(series: pd.Series, period: int = 30,
                          annualize: bool = True) -> pd.Series:
    """Historical Volatility (annualized by default)"""
    log_ret = np.log(series / series.shift(1))
    vol = log_ret.rolling(window=period, min_periods=period).std()
    if annualize:
        vol = vol * np.sqrt(365)  # untuk crypto 24/7
    return vol


# ==========================================
# 🧭 TREND INDICATORS
# ==========================================

def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index - mengukur kekuatan trend
    ADX > 25 = trend kuat, ADX < 20 = sideways
    Return DataFrame: adx, plus_di, minus_di
    """
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_val = atr(high, low, close, period)

    plus_di = 100 * ema(plus_dm, period) / atr_val
    minus_di = 100 * ema(minus_dm, period) / atr_val

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = ema(dx, period)

    return pd.DataFrame({
        'adx': adx_val,
        'plus_di': plus_di,
        'minus_di': minus_di
    }, index=close.index)


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Supertrend - trend following indicator dengan stop-loss built-in
    Return DataFrame: supertrend, direction (1=bullish, -1=bearish)
    """
    atr_val = atr(high, low, close, period)

    hl_avg = (high + low) / 2
    upper_band = hl_avg + (multiplier * atr_val)
    lower_band = hl_avg - (multiplier * atr_val)

    supertrend_val = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    for i in range(period, len(close)):
        if i == period:
            supertrend_val.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
            continue

        prev_st = supertrend_val.iloc[i-1]
        prev_close = close.iloc[i-1]
        curr_close = close.iloc[i]

        # Lower band
        if lower_band.iloc[i] > prev_st or prev_close < prev_st:
            new_st = lower_band.iloc[i]
        else:
            new_st = prev_st

        # Upper band
        if upper_band.iloc[i] < prev_st or prev_close > prev_st:
            new_st_upper = upper_band.iloc[i]
        else:
            new_st_upper = prev_st

        if curr_close > new_st_upper:
            supertrend_val.iloc[i] = new_st
            direction.iloc[i] = 1
        else:
            supertrend_val.iloc[i] = new_st_upper
            direction.iloc[i] = -1

    return pd.DataFrame({
        'supertrend': supertrend_val,
        'direction': direction
    }, index=close.index)


# ==========================================
# 📊 VOLUME INDICATORS
# ==========================================

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume - kumulatif volume berdasarkan arah harga"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    """
    Volume Weighted Average Price (untuk intraday)
    Note: Untuk daily data, ini adalah rolling VWAP
    """
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).rolling(window=20).sum()
    cum_vol = volume.rolling(window=20).sum()
    return cum_tp_vol / cum_vol


# ==========================================
# 🎯 HELPER FUNCTIONS
# ==========================================

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan SEMUA indikator ke DataFrame sekaligus.
    Input: df dengan kolom [open, high, low, close, volume]
    Output: df dengan semua kolom indikator tambahan
    """
    df = df.copy()

    # Moving Averages
    df['sma_20'] = sma(df['close'], 20)
    df['sma_50'] = sma(df['close'], 50)
    df['sma_200'] = sma(df['close'], 200)
    df['ema_12'] = ema(df['close'], 12)
    df['ema_26'] = ema(df['close'], 26)

    # Momentum
    df['rsi_14'] = rsi(df['close'], 14)
    macd_df = macd(df['close'])
    df = pd.concat([df, macd_df], axis=1)
    stoch_df = stochastic(df['high'], df['low'], df['close'])
    df = pd.concat([df, stoch_df], axis=1)

    # Volatility
    bb_df = bollinger_bands(df['close'])
    df = pd.concat([df, bb_df], axis=1)
    df['atr_14'] = atr(df['high'], df['low'], df['close'], 14)
    df['hist_vol_30'] = historical_volatility(df['close'], 30)

    # Trend
    adx_df = adx(df['high'], df['low'], df['close'])
    df = pd.concat([df, adx_df], axis=1)
    st_df = supertrend(df['high'], df['low'], df['close'])
    df = pd.concat([df, st_df], axis=1)

    # Volume
    df['obv'] = obv(df['close'], df['volume'])

    return df


def normalize_for_ai(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Normalize indikator ke skala 0-1 untuk input AI.
    Gunakan min-max scaling dengan rolling window.
    """
    df = df.copy()
    window = 100  # rolling window untuk normalisasi

    for col in columns:
        if col in df.columns:
            rolling_min = df[col].rolling(window=window, min_periods=1).min()
            rolling_max = df[col].rolling(window=window, min_periods=1).max()
            df[f'{col}_norm'] = (df[col] - rolling_min) / (rolling_max - rolling_min + 1e-8)

    return df