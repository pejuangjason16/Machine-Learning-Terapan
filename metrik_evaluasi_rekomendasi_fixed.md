# Metrik Evaluasi dalam Sistem Rekomendasi

Dokumen ini menjelaskan metrik-metrik evaluasi yang digunakan untuk mengukur kinerja dua jenis sistem rekomendasi yang dibangun: **Content-Based Filtering** dan **Collaborative Filtering (SVD)**.

---

## 1. Cosine Similarity

### 📌 Digunakan untuk:
Mengukur kemiripan antar review dalam Content-Based Filtering, khususnya teks ulasan pengguna.

### 📐 Rumus:

$$
\text{cos}(\theta) = \frac{A \cdot B}{\|A\| \cdot \|B\|}
$$

### 🧠 Penjelasan:
- **A** dan **B** adalah dua vektor representasi TF-IDF dari review.
- **A · B** adalah hasil dot product antara vektor A dan B.
- **||A||** dan **||B||** adalah panjang (norma) masing-masing vektor.
- Nilai cosine similarity berkisar antara 0 dan 1.
- Semakin mendekati 1, semakin mirip dua review secara semantik.

---

## 2. RMSE (Root Mean Squared Error)

### 📌 Digunakan untuk:
Mengukur akurasi prediksi rating dalam Collaborative Filtering (SVD).

### 📐 Rumus:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{r}_i - r_i)^2}
$$

### 🧠 Penjelasan:
- **hat_r_i**: rating yang diprediksi oleh model.
- **r_i**: rating asli dari pengguna.
- **n**: jumlah data prediksi yang dievaluasi.
- RMSE menunjukkan rata-rata selisih kuadrat antara prediksi dan kenyataan.
- Semakin kecil RMSE, semakin akurat prediksi model.

---

## ✨ Kesimpulan:
- **Cosine Similarity** digunakan untuk mengevaluasi seberapa mirip dua review berdasarkan isi teks.
- **RMSE** digunakan untuk mengevaluasi seberapa tepat model memprediksi rating numerik.
- Keduanya saling melengkapi dalam mengevaluasi sistem rekomendasi.
