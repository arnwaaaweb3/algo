# 📄 Dokumentasi: BTC Data Fetcher (Yahoo Finance)

## 🎯 Fungsi Utama
File ini bertugas sebagai **Data Pipeline tahap awal**. Fungsinya adalah mengunduh data historis harga Bitcoin (BTC/USD) dari Yahoo Finance, merapikan format datanya, dan menyimpannya ke dalam file CSV. 

Data yang dihasilkan dari skrip ini akan menjadi fondasi untuk tahap selanjutnya: *Backtesting*, *Paper Trading*, hingga *Agentic AI*.

---

## ⚙️ Cara Kerja (How it Works)

Skrip ini menjalankan fungsi `fetch_btc_yfinance()` dengan alur kerja sebagai berikut:

1. **Mengambil Data Mentah:** 
   Terkoneksi ke Yahoo Finance menggunakan *library* `yfinance` dan menarik data *candlestick* (Open, High, Low, Close, Volume) berdasarkan interval waktu yang diminta (misal: harian `1d` atau per jam `1h`).

2. **Standarisasi Nama Kolom:** 
   Mengubah semua nama kolom menjadi huruf kecil (`open`, `high`, `low`, `close`, `volume`). 
   *Kenapa?* Karena hampir semua *engine backtesting* dan *library AI* menuntut format kolom standar huruf kecil. Ini menghemat waktu kita di masa depan.

3. **Pembersihan Format Waktu:** 
   Menghapus informasi *timezone* dari kolom tanggal/waktu (`tz_localize(None)`). Ini mencegah *error* saat data diolah lebih lanjut menggunakan `pandas`.

4. **Menjadikan Waktu sebagai Index:** 
   Kolom `datetime` dijadikan *index* utama. Ini adalah standar emas untuk data finansial (*time-series*) agar operasi seperti pergeseran waktu (*shifting*) atau penggabungan data bisa berjalan sangat cepat.

5. **Penyimpanan Otomatis:** 
   Data yang sudah bersih disimpan otomatis ke dalam folder `data/` dengan nama file `btc_usdt_{interval}.csv`.

---

## 📂 Struktur Output (File CSV)

Setelah skrip dijalankan, akan muncul file CSV di folder `data/` dengan struktur kolom seperti ini:

| datetime (Index)      | open   | high   | low    | close  | volume   |
|-----------------------|--------|--------|--------|--------|----------|
| 2018-01-01 00:00:00   | 13850  | 14500  | 13500  | 14200  | 15000.5  |
| 2018-01-02 00:00:00   | 14200  | 15000  | 14000  | 14800  | 18000.2  |

---

## 🚀 Cara Menjalankan

Pastikan kamu sudah menginstall *library* yang dibutuhkan:
```bash
pip install pandas yfinance
```

Jalankan skrip dari terminal/command prompt:
```bash
python fetch_data.py
```
