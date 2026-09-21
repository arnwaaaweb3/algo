# 📊 Dokumentasi: EDA & Visualisasi Data BTC

## 🎯 Fungsi Utama
File ini bertugas melakukan **Exploratory Data Analysis (EDA)** pada data historis BTC yang sudah diunduh. Fungsinya adalah menghitung statistik penting (volatilitas, drawdown, return) dan membuat visualisasi 3-panel untuk memahami karakter pasar BTC sebelum membangun strategi trading.

Output dari skrip ini akan menjadi **dasar pengambilan keputusan** untuk parameter *risk management* dan *feature engineering* di tahap selanjutnya.

---

## ⚙️ Cara Kerja (How it Works)

Skrip ini menjalankan alur kerja sebagai berikut:

### 1. Normalisasi Schema Data
Mendeteksi dan menyeragamkan nama kolom tanggal dari berbagai format (bisa `datetime`, `date`, atau `open_time`) menjadi satu kolom standar bernama `"date"`. Ini memastikan skrip tetap jalan meski sumber data berubah-ubah.

### 2. Perhitungan Return Harian
Menghitung persentase perubahan harga penutupan dari hari sebelumnya menggunakan `pct_change()`. Ini adalah dasar untuk menghitung volatilitas dan metrik risiko lainnya.

### 3. Perhitungan Statistik Kunci
Skrip menghitung 4 metrik penting:
- **Annualized Volatility (`ann_vol`):** Seberapa fluktuatif harga BTC dalam setahun. Angka tinggi = risiko tinggi.
- **Maximum Drawdown (`max_dd`):** Penurunan terdalam dari puncak ke lembah. Ini menunjukkan risiko terburuk yang pernah terjadi.
- **Total Return:** Persentase keuntungan/kerugian dari awal periode sampai akhir.
- **Skewness:** Mengukur ketimpangan distribusi return. Nilai positif = lebih sering ada lonjakan harga tinggi.

### 4. Visualisasi 3-Panel
Membuat grafik bertumpuk yang menampilkan:
- **Panel Atas:** Tren harga BTC dari waktu ke waktu.
- **Panel Tengah:** Return harian dalam bentuk bar chart (hijau = naik, merah = turun).
- **Panel Bawah:** Drawdown (persentase penurunan dari puncak) dalam bentuk area chart.

### 5. Penyimpanan Output
Grafik disimpan otomatis ke file `data/exploration.png` dengan resolusi 110 DPI.

---

## 📤 Output yang Dihasilkan

### 1. Statistik di Terminal
```bash
baris : 3181
periode : 2018-01-01 -> 2026-09-16
harga terakhir : $65,432
total return : +372.45%
volatilitas ann.: 68.3%
max drawdown : -77.2%
skewness return : 0.45 (>0 = ekor kanan tebal)
```
### 2. File Gambar
- **Lokasi:** `data/exploration.png`
- **Isi:** 3 grafik bertumpuk (Harga, Return Harian, Drawdown)

---

## 🚀 Cara Menjalankan

Pastikan kamu sudah menginstall *library* yang dibutuhkan:
```bash
pip install pandas numpy matplotlib
```
Jalankan skrip dari terminal/command prompt:
```bash
python explore_data.py
```

