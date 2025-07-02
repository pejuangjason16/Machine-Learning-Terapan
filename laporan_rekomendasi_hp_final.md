# Laporan Proyek Machine Learning - Nama Anda

## Project Overview

Buku dan film Harry Potter telah menjadi fenomena global selama lebih dari dua dekade. Dengan basis penggemar yang sangat besar, berbagai platform kini menyediakan tempat bagi pembaca dan penonton untuk mengekspresikan pendapat mereka melalui ulasan dan rating. Data ulasan ini sangat berguna untuk memahami persepsi publik terhadap setiap buku dalam seri Harry Potter.

Dalam proyek ini, dilakukan analisis eksploratif dan pembangunan sistem rekomendasi berbasis ulasan pengguna terhadap buku Harry Potter. Tujuan proyek adalah menggali insight dari review pengguna dan mengembangkan model rekomendasi berdasarkan sentimen atau rating yang diberikan.

## Business Understanding

### Problem Statements

- Buku Harry Potter mana yang paling disukai berdasarkan data ulasan?
- Apakah terdapat pola tertentu antara jumlah likes dan rating bintang (stars_given)?
- Bagaimana membuat sistem rekomendasi buku berdasarkan ulasan atau penilaian?

### Goals

- Menganalisis distribusi dan tren ulasan pengguna.
- Mengidentifikasi buku favorit berdasarkan data rating.
- Mengembangkan sistem rekomendasi berbasis konten ulasan atau fitur numerik.

### Solution statements

- Menggunakan teknik eksplorasi data untuk melihat pola rating dan popularitas tiap buku.
- Mengembangkan sistem content-based recommendation berbasis kolom `description`.
- Menggunakan algoritma TF-IDF dan Cosine Similarity untuk menghitung kemiripan antar review.

## Data Understanding

Dataset berisi 720 entri ulasan terhadap buku-buku Harry Potter dengan kolom sebagai berikut:

- `book`: Judul buku Harry Potter yang direview.
- `name`: Nama reviewer.
- `date`: Tanggal review dalam format seperti 'Nov11,2019'.
- `rating`: Kategori rating yang diberikan (misalnya "liked it", "it was amazing", dll.).
- `likes`: Jumlah likes dari ulasan (berupa string seperti "1,234 likes", perlu dibersihkan).
- `description`: Isi ulasan (teks bebas).
- `stars_given`: Rating numerik antara 1.0 hingga 5.0.

Terdapat beberapa nilai yang hilang pada kolom `stars_given`, `rating`, dan `description`.

## Data Preparation

- Menghapus kolom `Unnamed: 0` karena tidak memiliki informasi penting.
- Membersihkan kolom `likes` dari string " likes" dan tanda koma, lalu ubah ke integer.
- Mengonversi kolom `date` ke format datetime menggunakan format spesifik `%b%d,%Y`.
- Tangani nilai null pada `stars_given` dan `description` jika diperlukan.
- Tambahkan kolom `month` dari `date` untuk analisis tren waktu.

```python
import pandas as pd

df = pd.read_csv("Film Harry Potter.csv")

# Hapus kolom tidak relevan
df.drop(columns=['Unnamed: 0'], inplace=True)

# Bersihkan kolom 'likes'
df['likes'] = df['likes'].str.replace(' likes', '', regex=False)
df['likes'] = df['likes'].str.replace(',', '', regex=False).astype(int)

# Konversi tanggal dengan format eksplisit
df['date'] = pd.to_datetime(df['date'], format='%b%d,%Y', errors='coerce')

# Tambahkan kolom bulan
df['month'] = df['date'].dt.to_period('M')

# Cek dan tangani missing value
print(df.isnull().sum())
```

## Modeling

### Content-Based Filtering dari Deskripsi

Menggunakan `TfidfVectorizer` untuk membangun model rekomendasi berdasarkan kemiripan antar review `description`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Hapus NaN di kolom deskripsi
df = df.dropna(subset=['description'])

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description'])

cos_sim = cosine_similarity(tfidf_matrix)

# Fungsi rekomendasi berbasis deskripsi
def recommend_by_description(index, cosine_sim=cos_sim):
    scores = list(enumerate(cosine_sim[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]
    return df.iloc[[i[0] for i in scores]][['book', 'name', 'stars_given']]

recommend_by_description(10)  # Contoh rekomendasi dari review ke-10
```

## Evaluation

- Dilakukan evaluasi manual dengan melihat seberapa relevan rekomendasi yang diberikan dengan review asli.
- Pengamatan visual menunjukkan bahwa review dengan isi dan bintang yang mirip direkomendasikan bersama.
- Rekomendasi yang dihasilkan dengan Cosine Similarity menunjukkan kecenderungan memilih review dengan deskripsi sentiment serupa.

**---Ini adalah bagian akhir laporan---**
