
# Laporan Proyek Machine Learning - Jason Hendrawan

## Domain Proyek

Pasar modal, khususnya pasar saham, merupakan ekosistem yang kompleks dan dinamis, dipengaruhi oleh berbagai faktor ekonomi makro, kebijakan pemerintah, peristiwa geopolitik, dan sentimen investor. Fluktuasi harga saham harian dapat menghadirkan peluang keuntungan yang signifikan sekaligus risiko kerugian yang substansial bagi para pelaku pasar. Oleh karena itu, kemampuan untuk memahami dan memprediksi pergerakan harga saham merupakan tantangan fundamental yang terus menjadi fokus penelitian dalam analisis keuangan.

Proyek ini secara spesifik berfokus pada analisis dan prediksi harga saham PT Aneka Tambang Tbk (ANTM), sebuah perusahaan pertambangan terkemuka di Indonesia. Sebagai entitas yang beroperasi di sektor komoditas, harga saham ANTM sangat rentan terhadap dinamika pasar global dan domestik, termasuk perubahan harga komoditas, kebijakan ekspor/impor, dan kondisi ekonomi global. Karakteristik ini menjadikan ANTM sebagai studi kasus yang menarik dan relevan untuk aplikasi teknik machine learning dalam prediksi pasar.

Kemampuan untuk memprediksi arah pergerakan harga saham, bahkan dengan tingkat akurasi yang moderat, dapat memberikan keunggulan kompetitif yang signifikan bagi investor dan analis keuangan. Hal ini memungkinkan perumusan keputusan investasi yang lebih terinformasi, implementasi strategi manajemen risiko yang lebih efektif, dan potensi optimalisasi keuntungan. Pendekatan analisis fundamental atau teknikal tradisional, meskipun penting, seringkali tidak cukup untuk menangkap semua nuansa pasar yang kompleks dan non-linear. Oleh karena itu, diperlukan metode yang lebih canggih untuk mengidentifikasi pola tersembunyi dan hubungan yang tidak linear dalam data pasar.

Machine learning menawarkan pendekatan yang kuat untuk mengatasi kompleksitas ini. Dengan melatih model pada data historis harga dan volume perdagangan, sistem dapat dibangun untuk mengidentifikasi pola yang mungkin tidak terlihat oleh metode statistik konvensional. Pendekatan ini memungkinkan pengembangan sistem yang mampu mengklasifikasikan atau memprediksi arah pergerakan harga saham di masa depan, memberikan sinyal yang dapat digunakan sebagai dasar pengambilan keputusan.

## Tujuan Proyek

Tujuan utama dari proyek ini adalah membangun model machine learning yang dapat mengklasifikasikan arah pergerakan harga saham PT Aneka Tambang Tbk (ANTM) pada hari perdagangan berikutnya. Untuk mencapai tujuan ini, dilakukan eksplorasi data historis, pembentukan fitur teknikal seperti Moving Average, MACD, dan RSI, serta pemodelan menggunakan beberapa algoritma klasifikasi.

## Dataset dan Pra-Pemrosesan

Dataset yang digunakan adalah data historis harga saham ANTM yang mencakup kolom Tanggal, Terakhir, Pembukaan, Tertinggi, Terendah, Vol., dan Perubahan%. Dataset telah dibersihkan dengan:
- Konversi format harga dan volume ke numerik
- Parsing volume dengan satuan K/M/B
- Penghapusan baris kosong dan konversi tipe data

## Feature Engineering

Fitur teknikal berikut telah ditambahkan:
- Lagged features: `Terakhir_t-1`, `Vol_t-1`, `Perubahan%_t-1`
- Moving Average: `MA5`, `MA10`
- Exponential Moving Average: `EMA12`, `EMA26`
- MACD dan MACD signal line
- RSI14 (Relative Strength Index)
- Target biner (`Target`): 1 jika harga naik esok hari, 0 jika tidak

## Modeling dan Evaluasi

Tiga model dikembangkan:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

Setiap model menjalani **hyperparameter tuning** menggunakan `GridSearchCV` dengan validasi silang. Hasil evaluasi dilakukan pada data uji (test set).

### Hasil Evaluasi:

**Logistic Regression**
- Akurasi: ~58.9%
- Recall kelas 1 (naik): 0.00 (model tidak mendeteksi kenaikan)
- F1-score kelas 1: 0.00

**SVM**
- Akurasi: ~58.9%
- Recall kelas 1: 0.00
- F1-score kelas 1: 0.00

**Random Forest**
- Akurasi: ~57%
- Recall kelas 1: 0.24
- Precision kelas 1: 0.45
- F1-score kelas 1: 0.32

### Visualisasi Hasil:

**Confusion Matrix** (Heatmap)
- Menunjukkan jumlah benar/salah klasifikasi untuk setiap kelas
- Logistic Regression dan SVM hanya memprediksi satu kelas
- Random Forest mulai mengenali kedua kelas

**ROC Curve dan AUC**
- ROC Curve dibuat untuk semua model
- Random Forest memiliki AUC tertinggi (~0.75), menunjukkan trade-off TPR/FPR terbaik

## Kesimpulan

- Random Forest menghasilkan performa terbaik meskipun belum optimal
- Model sangat sensitif terhadap distribusi tidak seimbang (imbalance class)
- Model baseline cenderung overpredict satu kelas (turun/stagnan)
- Visualisasi ROC dan Confusion Matrix sangat membantu dalam analisis lanjutan

## Saran Perbaikan

1. Tambahkan teknik **class balancing** (SMOTE, class_weight)
2. Gunakan lebih banyak **indikator teknikal lanjutan** (Bollinger Bands, Stochastic)
3. Lanjutkan dengan **backtesting strategi** jika model digunakan dalam pengambilan keputusan nyata
4. Integrasikan dengan **sentimen analysis** berbasis teks berita/berita sosial

## Referensi

1. Harga-Emas.org, “Grafik Harga Emas Hari Ini,” https://harga-emas.org/grafik  
2. E. F. Fama, “Efficient Capital Markets: A Review of Theory and Empirical Work,” The Journal of Finance, vol. 25, no. 2, pp. 383–417, 1970.  
3. Investing.com, “Aneka Tambang Tbk (ANTM) Historical Data,” 2025. https://id.investing.com/equities/aneka-tambang-historical-data
