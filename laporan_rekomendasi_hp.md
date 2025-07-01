# Laporan Proyek Machine Learning - Nama Anda

## Project Overview

Film merupakan salah satu bentuk hiburan yang sangat populer di kalangan masyarakat. Di antara berbagai waralaba film, Harry Potter menempati posisi istimewa karena memiliki basis penggemar yang besar dan konsisten. Namun, tidak semua film atau karakter dalam seri tersebut mendapatkan respon atau rating yang sama dari penonton.

Dengan semakin banyaknya data yang tersedia terkait film, seperti rating, pendapatan, dan jumlah penonton, muncul kebutuhan untuk mengembangkan sistem rekomendasi yang dapat membantu pengguna menemukan film yang paling sesuai dengan preferensinya.

Dalam proyek ini, akan dikembangkan sistem rekomendasi berbasis machine learning menggunakan data dari film-film Harry Potter. Tujuan akhirnya adalah memberikan rekomendasi film terbaik berdasarkan rating, popularitas, dan metrik lainnya.

## Business Understanding

### Problem Statements

- Bagaimana mengidentifikasi film Harry Potter terbaik berdasarkan rating dan pendapatan?
- Bagaimana menyusun sistem rekomendasi film berbasis konten (content-based recommendation)?
- Bagaimana mengevaluasi kualitas sistem rekomendasi tersebut?

### Goals

- Menentukan film-film terbaik berdasarkan data historis.
- Membangun sistem rekomendasi untuk menyarankan film berdasarkan kesamaan fitur (durasi, genre, rating).
- Mengevaluasi sistem rekomendasi menggunakan metrik evaluasi yang relevan.

### Solution statements

- Menggunakan algoritma **Cosine Similarity** dalam pendekatan Content-Based Filtering.
- Membandingkan hasil rekomendasi dengan pendekatan kedua menggunakan **K-Nearest Neighbors (KNN)** berbasis fitur numerik.

## Data Understanding

Dataset berisi informasi mengenai 8 film dari seri Harry Potter dengan atribut sebagai berikut:

- `Judul`: Nama film.
- `Tanggal Rilis`: Tanggal film dirilis.
- `Durasi`: Durasi film dalam menit.
- `Genre`: Genre film (fantasy, adventure, dll.).
- `Rating`: Rating film dari IMDb.
- `Pendapatan`: Pendapatan global (dalam juta USD).
- `Jumlah Penonton`: Estimasi jumlah penonton.
- `Deskripsi`: Ringkasan isi cerita film.

Contoh data:
| Judul | Durasi | Rating | Pendapatan |
|-------|--------|--------|-------------|
| Harry Potter and the Sorcerer's Stone | 152 | 7.6 | 974.8 |

Terdapat 8 entri (film), seluruhnya dari waralaba Harry Potter.

## Data Preparation

Langkah-langkah persiapan data meliputi:

- Konversi kolom `Tanggal Rilis` ke format datetime.
- Menghapus simbol `$` dan `,` pada kolom `Pendapatan`, lalu mengubahnya ke numerik.
- Normalisasi fitur numerik untuk modeling (durasi, rating, pendapatan).
- Menggabungkan kolom teks (`Deskripsi`, `Genre`) untuk content-based filtering.

Contoh kode:
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Film Harry Potter.csv')

# Konversi tanggal rilis
df['Tanggal Rilis'] = pd.to_datetime(df['Tanggal Rilis'])

# Bersihkan pendapatan
df['Pendapatan'] = df['Pendapatan'].replace('[\$,]', '', regex=True).astype(float)

# Normalisasi fitur
scaler = MinMaxScaler()
df[['Durasi', 'Rating', 'Pendapatan']] = scaler.fit_transform(df[['Durasi', 'Rating', 'Pendapatan']])
```

## Modeling

### Content-Based Filtering (Cosine Similarity)

Menggunakan `TfidfVectorizer` pada kolom `Deskripsi` + `Genre` untuk membuat vektor teks.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df['combined'] = df['Deskripsi'] + " " + df['Genre']
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])

cos_sim = cosine_similarity(tfidf_matrix)

# Fungsi rekomendasi
def recommend(title, cosine_sim=cos_sim):
    idx = df[df['Judul'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]
    film_indices = [i[0] for i in scores]
    return df['Judul'].iloc[film_indices]

recommend("Harry Potter and the Sorcerer's Stone")
```

### KNN Recommender

Menggunakan fitur numerik: `Durasi`, `Rating`, `Pendapatan`.

```python
from sklearn.neighbors import NearestNeighbors

features = df[['Durasi', 'Rating', 'Pendapatan']]
model_knn = NearestNeighbors(metric='euclidean', algorithm='auto')
model_knn.fit(features)

distances, indices = model_knn.kneighbors(features, n_neighbors=4)

# Fungsi rekomendasi
def knn_recommend(title):
    idx = df[df['Judul'] == title].index[0]
    recs = indices[idx][1:]
    return df['Judul'].iloc[recs]

knn_recommend("Harry Potter and the Sorcerer's Stone")
```

## Evaluation

Karena ini sistem rekomendasi, kita menggunakan evaluasi berbasis kemiripan dan logika domain. Beberapa metrik umum seperti **Precision@K** atau **Silhouette Score** untuk clustering tidak berlaku langsung tanpa ground truth. Oleh karena itu, evaluasi dilakukan berdasarkan:

- Apakah film yang direkomendasikan berada di satu waralaba dan memiliki rating yang tinggi?
- Seberapa mirip genre dan deskripsi film yang direkomendasikan?

Untuk validasi tambahan, bisa digunakan *manual check* atau rating dari IMDb sebagai dasar pembobotan film yang direkomendasikan.

**---Ini adalah bagian akhir laporan---**
