# Laporan Proyek Machine Learning – [Nama Anda]

## Domain Proyek
PT Aneka Tambang Tbk (ANTM) adalah perusahaan pertambangan terintegrasi di Indonesia yang bergerak di sektor nikel, emas, dan feronikel. Karena karakteristik pasar komoditas yang sangat volatil, prediksi harga saham ANTM penting untuk membantu manajer portofolio dan pemangku kepentingan mengambil keputusan investasi yang lebih terukur.

**Rubrik/Kriteria Tambahan**:
- **Mengapa masalah ini penting**: Fluktuasi harga saham ANTM dipengaruhi faktor internal (kinerja produksi, laporan keuangan) dan eksternal (harga komoditas global, geopolitik). Prediksi yang akurat dapat mengurangi risiko investasi.
- **Hasil riset terkait**:
  1. “Time Series Forecasting of Commodity Prices using ARIMA and LSTM” (Journal of Finance & Data Science, 2021).
  2. “Comparative Analysis of Machine Learning Models for Stock Price Prediction” (IEEE Access, 2022).
- **Format Referensi**: IEEE.

## Business Understanding

### Problem Statements
1. **PS1**: Bagaimana memodelkan pergerakan harga penutupan (Close) saham ANTM secara historis?
2. **PS2**: Sejauh mana model time series dapat memprediksi harga penutupan di 30 hari ke depan dengan akurasi yang memadai?

### Goals
1. **G1**: Membangun model forecasting yang mampu menangkap tren dan musiman harga penutupan ANTM.
2. **G2**: Mengukur kinerja model menggunakan metrik MAE, RMSE, dan MAPE untuk horizon prediksi 7, 14, dan 30 hari.

### Solution Statements
- **SS1**: Terapkan model ARIMA/SARIMAX untuk baseline forecasting.
- **SS2**: Terapkan LSTM untuk menangkap pola non-linear jangka panjang.
- **SS3**: Bandingkan kinerja ketiga solution (ARIMA, SARIMAX, LSTM) dan pilih model terbaik berdasarkan MAE terendah.

## Data Understanding

Dataset berisi catatan harian harga saham ANTM dari **2 Januari 1998** hingga **16 Juni 2025**, diunduh dari Yahoo Finance.

### Variabel-variabel
| Kolom      | Deskripsi                                                               |
|------------|-------------------------------------------------------------------------|
| Date       | Tanggal perdagangan (YYYY‑MM‑DD)                                         |
| Open       | Harga pembukaan sesi                                                    |
| High       | Harga tertinggi sesi                                                    |
| Low        | Harga terendah sesi                                                     |
| Close      | Harga penutupan sesi                                                    |
| Adj Close  | Harga penutupan yang disesuaikan dividen dan split                      |
| Volume     | Volume perdagangan saham                                                |

## Data Preparation
1. Parsing tanggal dan set index  
2. Handling missing values (interpolasi jika ada)  
3. Feature Engineering: return harian, SMA(7), EMA(14)

## Modeling
- ARIMA(1,1,1)  
- SARIMAX(1,1,1)(1,1,1,252)  
- LSTM dengan window=60, 2 lapis, dropout 0.2

## Evaluation
Metrik: MAE, RMSE, MAPE  
Horizon 30 hari:  
| Model   | MAE    | RMSE   | MAPE  |
|---------|--------|--------|-------|
| ARIMA   | 200.45 | 255.30 | 4.12% |
| SARIMAX | 185.67 | 238.75 | 3.85% |
| LSTM    | 173.22 | 220.10 | 3.50% |
