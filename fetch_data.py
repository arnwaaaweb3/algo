import pandas as pd
import yfinance as yf
import os
import logging
from datetime import datetime, timedelta

# ==========================================
# 1. SETUP LOGGING (Pengganti print)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(".log"), # Simpan log ke file
        logging.StreamHandler()                  # Tetap tampilkan di layar
    ]
)

def get_last_saved_datetime(filename):
    """Cek waktu candle terakhir yang sudah ada di CSV."""
    if not os.path.exists(filename):
        return None
    
    try:
        df_existing = pd.read_csv(filename, parse_dates=['datetime'], index_col='datetime')
        if df_existing.empty:
            return None
        return df_existing.index.max()
    except Exception as e:
        logging.warning(f"Gagal membaca file lama: {e}. Akan download ulang dari awal.")
        return None

def fetch_btc_yfinance(interval='1d', start_str='2018-01-01'):
    """
    Mengunduh data historis BTC/USD. 
    V2.0: Support incremental update & validasi data bolong.
    """
    filename = f"data/btc_usdt_{interval}.csv"
    os.makedirs('data', exist_ok=True)
    
    # Cek apakah ini update pertama kali atau update bertahap
    last_dt = get_last_saved_datetime(filename)
    
    if last_dt is not None:
        # Kalau sudah ada data, mulai download dari 1 interval setelah data terakhir
        # (Kita pakai timedelta sederhana, yfinance akan otomatis handle overlap)
        start_str = last_dt.strftime('%Y-%m-%d')
        logging.info(f"[-] Mode Incremental: Melanjutkan data dari {start_str}")
    else:
        logging.info(f"[-] Mode Full Download: Mengambil data dari {start_str}")

    try:
        ticker = yf.Ticker("BTC-USD")
        df_new = ticker.history(start=start_str, interval=interval)
        
        if df_new.empty:
            logging.warning("[!] Tidak ada data baru dari Yahoo Finance.")
            return None

        # --- STANDARDISASI (Sama seperti kodemu sebelumnya) ---
        df_new.reset_index(inplace=True)
        date_col = 'Date' if 'Date' in df_new.columns else 'Datetime'
        df_new.rename(columns={
            date_col: 'datetime', 'Open': 'open', 'High': 'high', 
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }, inplace=True)
        
        df_new = df_new[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        df_new['datetime'] = pd.to_datetime(df_new['datetime']).dt.tz_localize(None)
        
        # --- FITUR BARU: GABUNGKAN DENGAN DATA LAMA (INCREMENTAL) ---
        if last_dt is not None:
            df_old = pd.read_csv(filename, parse_dates=['datetime'], index_col='datetime')
            df_combined = pd.concat([df_old, df_new.set_index('datetime')])
            # Hapus duplikat kalau ada overlap waktu
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            df_combined.sort_index(inplace=True)
        else:
            df_combined = df_new.set_index('datetime')

        # --- FITUR BARU: VALIDASI CANDLE BOLONG (MISSING DATA) ---
        # Crypto market 24/7, jadi tidak boleh ada candle yang bolong.
        freq_map = {'1d': 'D', '1h': 'h', '1m': 'min'}
        freq = freq_map.get(interval, 'D')
        
        full_range = pd.date_range(start=df_combined.index.min(), end=df_combined.index.max(), freq=freq)
        missing_candles = full_range.difference(df_combined.index)
        
        if len(missing_candles) > 0:
            logging.warning(f"[!] PERINGATAN: Ditemukan {len(missing_candles)} candle bolong! "
                            f"Contoh: {missing_candles[:3].tolist()}")
            # Opsional: isi candle yang bolong dengan NaN atau forward-fill
            df_combined = df_combined.reindex(full_range)
            df_combined.ffill(inplace=True) # Isi kekosongan dengan harga sebelumnya
        else:
            logging.info("[+] Validasi sukses: Tidak ada candle yang bolong (Data 100% continuous).")

        # Simpan ke CSV
        df_combined.to_csv(filename)
        logging.info(f"[+] Total candle saat ini: {len(df_combined)} | Saved to {filename}\n")
        
        return df_combined

    except Exception as e:
        logging.error(f"[!] Error fetching data via yfinance: {e}")
        return None

if __name__ == "__main__":
    logging.info("=== Memulai Proses Fetching Data BTC ===")
    
    # Jalankan untuk Daily dan Hourly
    fetch_btc_yfinance(interval='1d', start_str='2018-01-01')
    fetch_btc_yfinance(interval='1h', start_str='2024-10-01')
    
    logging.info("=== Proses Selesai ===")