# Metrik Evaluasi dalam Sistem Rekomendasi

Dokumen ini menjelaskan metrik-metrik evaluasi yang digunakan untuk mengukur kinerja tiga pendekatan sistem rekomendasi: **Content-Based Filtering**, **Collaborative Filtering (SVD)**, dan **evaluasi Top-K**.

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
- \(\hat{r}_i\): rating yang diprediksi oleh model.
- \(r_i\): rating asli dari pengguna.
- \(n\): jumlah data prediksi yang dievaluasi.
- RMSE menunjukkan rata-rata selisih kuadrat antara prediksi dan kenyataan.
- Semakin kecil RMSE, semakin akurat prediksi model.

---

## 3. Precision@K dan Recall@K

### 📌 Digunakan untuk:
Mengukur performa sistem dalam menyarankan item relevan di antara Top-K hasil rekomendasi.

### 📐 Rumus:

**Precision@K**:

$$
\text{Precision@K} = \frac{\text{Jumlah item relevan di Top-K}}{K}
$$

**Recall@K**:

$$
\text{Recall@K} = \frac{\text{Jumlah item relevan di Top-K}}{\text{Total item relevan}}
$$

### 🧠 Penjelasan:
- Digunakan pada evaluasi offline Content-Based Filtering berbasis asumsi bahwa review lain dari user yang sama adalah “relevan”.
- Precision@K mengukur **akurasi**: seberapa banyak rekomendasi yang benar dari total yang disarankan.
- Recall@K mengukur **cakupan**: seberapa besar porsi item relevan berhasil ditangkap oleh sistem.
- Nilai berkisar antara 0 dan 1, makin tinggi makin baik.

---

## ✨ Kesimpulan:
- **Cosine Similarity** mengevaluasi kemiripan konten antar review.
- **RMSE** mengevaluasi seberapa akurat model memprediksi rating numerik.
- **Precision@K dan Recall@K** memberikan insight pada kualitas dan cakupan rekomendasi Top-K.
- Kombinasi metrik ini membantu menilai performa sistem rekomendasi dari sisi konten, prediksi, dan personalisasi.
