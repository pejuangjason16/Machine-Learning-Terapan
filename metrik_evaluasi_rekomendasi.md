# Metrik Evaluasi dalam Sistem Rekomendasi

Dokumen ini menjelaskan metrik-metrik evaluasi yang digunakan untuk mengukur kinerja dua jenis sistem rekomendasi yang dibangun: **Content-Based Filtering** dan **Collaborative Filtering (SVD)**.

---

## 1. Cosine Similarity

### 📌 Digunakan untuk:
- Mengukur kemiripan antar review dalam **Content-Based Filtering**, khususnya teks ulasan pengguna.

### 📐 Rumus:

Cosine Similarity antara dua vektor A dan B:

\[
\text{cos}(\theta) = \frac{A \cdot B}{\|A\| \cdot \|B\|}
\]

Di mana:
- \( A \cdot B \) = hasil dot product antara dua vektor
- \( \|A\| \) dan \( \|B\| \) adalah norma (panjang) masing-masing vektor

### 🧠 Cara kerja:
- Nilai Cosine Similarity berkisar antara **0 dan 1**.
- Nilai yang lebih tinggi menunjukkan **kemiripan yang lebih besar** antara isi review.
- Dalam proyek ini, review-review diubah menjadi vektor TF-IDF, lalu dihitung cosine similarity-nya.

---

## 2. RMSE (Root Mean Squared Error)

### 📌 Digunakan untuk:
- Mengukur **akurasi prediksi rating** dalam **Collaborative Filtering (SVD)**

### 📐 Rumus:

\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{r}_i - r_i)^2}
\]

Di mana:
- \( \hat{r}_i \) = rating yang diprediksi model
- \( r_i \) = rating aktual
- \( n \) = jumlah prediksi

### 🧠 Cara kerja:
- RMSE mengukur **seberapa besar rata-rata kesalahan kuadrat** antara prediksi dan kenyataan.
- Semakin kecil nilai RMSE, semakin akurat model.
- Dalam skala 1–5, nilai RMSE di bawah 1 dianggap cukup baik.

---

## ✨ Kesimpulan:
- **Cosine Similarity** cocok digunakan untuk mengevaluasi kemiripan konten (teks).
- **RMSE** cocok digunakan untuk mengevaluasi akurasi prediksi rating numerik.
- Kombinasi keduanya memberikan gambaran menyeluruh atas performa sistem rekomendasi yang dibangun.
