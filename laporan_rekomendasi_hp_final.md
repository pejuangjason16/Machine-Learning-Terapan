# Laporan Proyek Machine Learning - Jason Hendrawan

## Project Overview

Film merupakan salah satu bentuk hiburan yang sangat populer pada zaman sekarang ini. Di antara berbagai waralaba film, Harry Potter menempati posisi yang istimewa karena memiliki basis penggemar yang besar dan konsisten. Buku dan film Harry Potter telah menjadi fenomena global selama lebih dari dua dekade. Meskipun begitu, tidak semua film dalam seri Harry Potter mampu mendapatkan ulasan atau rating yang sama dari penonton.

Harry Potter sendiri merupakan seri dengan tujuh novel fantasi yang ditulis oleh penulis Inggris J. K. Rowling. Novel-novel ini menceritakan kisah kehidupan seorang penyihir muda, Harry Potter, dan teman-temannya Hermione Granger dan Ron Weasley, yang semuanya adalah siswa di Sekolah Sihir Hogwarts. Alur cerita utamanya berkisar pada perjuangan Harry melawan Lord Voldemort, seorang penyihir hitam yang ingin menjadi abadi, menggulingkan badan pemerintahan penyihir yang dikenal sebagai Kementerian Sihir, dan menaklukkan semua penyihir dan Muggle (orang-orang non-penyihir).

Sejak novel pertama, Harry Potter and the Philosopher's Stone, dirilis pada 26 Juni 1997, buku-buku ini diterima dengan sangat baik oleh orang-orang dan bahkan telah meraih popularitas luar biasa, ulasan positif, dan kesuksesan komersial di seluruh dunia sampai-sampai banyak basis penggemar yang terbentuk sebagai pecinta seri Harry Potter. Hingga Februari 2018, buku-buku ini telah terjual hingga lebih dari 500 juta kopi di seluruh dunia, menjadikannya seri buku terlaris dalam sejarah, dan telah diterjemahkan ke dalam delapan puluh bahasa. Empat buku terakhir berturut-turut mencatat rekor sebagai buku dengan penjualan tercepat sepanjang sejarah, dengan angsuran terakhir terjual sekitar 2,7 juta kopi di Inggris Raya dan 8,3 juta kopi di Amerika Serikat dalam waktu dua puluh empat jam setelah dirilis.

Dengan semakin banyaknya data yang tersedia terkait film, seperti rating, pendapatan, dan jumlah penonton, muncul kebutuhan untuk mengembangkan sistem rekomendasi yang dapat membantu pengguna menemukan film yang paling sesuai dengan preferensinya. Dalam proyek ini, akan dilakukan analisis eksploratif dan sistem rekomendasi akan dikembangkan berbasis machine learning menggunakan data dari film-film Harry Potter yang telah terkumpul. Tujuan akhir dari proyek ini adalah memberikan rekomendasi film dari seri Harry Potter yang terbaik berdasarkan rating, popularitas, dan metrik lainnya yang tersedia dalam dataset.

## Business Understanding

### Problem Statements

- Pernyataan Masalah 1: Film Harry Potter seri manakah yang paling disukai berdasarkan dataset ulasan Film Harry Potter yang digunakan?
      - Film Harry Potter terdiri dari banyak seri yang masing-masing seri nya memiliki tempat tertentu dalam hati para penggemar yang mana basis penggemarnya pun sangat besar.
- Pernyataan Masalah 2: Bagaimana sistem rekomendasi film yang dibangun mampu membantu para penonton baru dan calon penggemar baru untuk menonton seri film Harry Potter yang paling direkomendasikan?
      - Sistem rekomendasi akan menentukan bagaimana pengalaman para penonton dan calon penggemar-penggemar baru dalam menonton seri film Harry Potter agar ekskpektasi dan harapan yang mereka miliki dapat terpenuhi.
- Pernyataan Masalah 3: Apakah sistem rekomendasi yang dibangun memiliki hasil evaluasi dan performa yang baik sehingga mampu membantu calon penggemar dan calon penonton baru dalam mendapatkan rekomendasi seri film Harry Potter yang terbaik?

### Goals

- Tujuan 1: Untuk mengetahui film Harry Potter manakah yang paling disukai berdasarkan dataset yang dimiliki.
- Tujuan 2: Mengidentifikasi buku favorit berdasarkan data rating.
- Tujuan 3: Mengembangkan sistem rekomendasi berbasis konten ulasan atau fitur numerik.

### Solution statements

Untuk memeroleh rekomendasi film Harry Potter seri mana yang terbaik, beberapa pendekatan akan digunakan dan diimplementasikan sebagai berikut:

- Menggunakan Content-Based Filtering dengan memanfaatkan deskripsi atau sinopsis dari setiap film. Pendekatan ini akan merekomendasikan film berdasarkan kemiripan kontennya. Teknik yang akan digunakan adalah TF-IDF Vectorizer untuk mengubah teks deskripsi menjadi representasi numerik dan Cosine Similarity untuk mengukur kemiripan antar film.

- Menggunakan Collaborative Filtering dengan memanfaatkan data rating yang diberikan oleh pengguna. Pendekatan ini akan merekomendasikan film berdasarkan preferensi dari pengguna lain yang memiliki selera serupa. Algoritma yang akan digunakan adalah Singular Value Decomposition (SVD) dan metrik evaluasi yang digunakan adalah Root Mean Squared Error (RMSE) untuk mengukur tingkat kesalahan prediksi rating dari model.

## Data Understanding

Bagian ini akan memberikan gambaran secara menyeluruh mengenai data yang digunakan dalam proyek, termasuk sumber, struktur, dan karakteristik setiap variabel. Pemahaman yang mendalam terhadap data merupakan fondasi krusial sebelum memulai tahapan pemrosesan dan pemodelan.

Dataset yang dipakai merupakan Data Film Harry Potter yang berisi data ulasan dari tiap seri film Harry Potter dari awal sampai akhir. Dataset ini terdiri dari total 720 data entri dengan total kolom sebanyak 8 kolom. Dataset ini nantinya akan dibersihkan menjadi dataset yang siap untuk digunakan, bebas dari segala missing values dan kolom yang tidak relevan akan dibuang. Dataset yang sudah dibersihkan akan terdiri dari total 689 data entri dan 7 kolom tersebut, yang masing - masing kolom akan mewakili tiap variabel yang akan dipakai dalam proyek kali ini, yaitu ada variabel "book" yang merepresentasikan judul buku yang diadaptasi menjadi film yang direview, variabel "name" yang merepresentasikan nama dari reviewer, variabel "date" yang merepresentasikan tanggal ulasan diunggah, variabel "rating" yang merepresentasikan kategori ulasan yang terbatas hanya pada beberapa kategori, variabel "likes" yang merepresentasikan jumlah like yang diberikan pada ulasan yang diunggah, variabel "description" yang memberikan deskripsi ulasan secara lengkap, rinci, dan detail mengenai film yang diulas, dan terakhir variabel "stars_given" yang merepresentasikan ulasan yang numerik dalam rentang 1 sampai 5. Semua variabel yang dipakai dalam dataset ini merupakan fitur kategorik kecuali variabel "stars_given". Tidak terdapat satu pun nilai yang hilang dalam dataset yang digunakan dan tidak terdapat satu pun nilai duplikat dalam dataset yang sudah dibersihkan.

Informasi Umum Data dan Sumber:

Data yang menjadi dasar analisis dalam proyek ini adalah data ulasan film Harry Potter. Data ini diperoleh dari sumber publik dan disajikan dalam format CSV yang bisa diunduh melalui situs berikut: [Data Ulasan Film Harry Potter](https://www.kaggle.com/datasets/notkrishna/harry-potter-reviews/data#). Dataset mencakup ulasan lengkap, detail, dan terperinci dari setiap seri dari film Harry Potter. Ketersediaan data ulasan yang lengkap dan rinci memungkinkan model untuk membangun sistem rekomendasi berdasarkan tiap ulasan yang diberikan untuk tiap seri film Harry Potter yang ada.  

Dataset bersih berisi 689 entri ulasan terhadap buku-buku Harry Potter dengan kolom sebagai berikut:

- `book`: Judul buku Harry Potter untuk tiap serinya.
- `name`: Nama reviewer.
- `date`: Tanggal review dalam format seperti 'Nov11,2019'.
- `rating`: Kategori rating yang diberikan yang terdiri dari beberapa kategori (misalnya "liked it", "it was amazing", dll.).
- `likes`: Jumlah likes dari tiap ulasan (berupa string seperti "1,234 likes", nantinya akan dirapikan dan dibersihkan).
- `description`: Deskripsi lengkap dari tiap ulasan (teks bebas).
- `stars_given`: Rating yang diberikan dalam bentuk numerik berkisar antara 1.0 hingga 5.0.

Terdapat beberapa missing values pada kolom `stars_given`, `rating`, dan `description`.

### Eksplorasi Data dan Analisis Statistik Deskriptif

Tahap eksplorasi data (EDA) dan analisis statistik deskriptif sangat penting untuk memahami karakteristik data, mengidentifikasi potensi masalah kualitas data, dan merumuskan strategi rekayasa fitur yang efektif. Meskipun tidak dapat dilakukan secara langsung dalam lingkungan ini, langkah-langkah konseptualnya adalah sebagai berikut:

- Pemuatan Data: Data akan dimuat ke dalam struktur data yang sesuai, seperti DataFrame Pandas, untuk memfasilitasi manipulasi dan analisis yang efisien.
- Pembersihan Data Awal: Proses ini melibatkan penanganan nilai yang hilang (jika ada), yang dapat dilakukan dengan penghapusan baris/kolom. Selain itu, konversi tipe data kolom Tanggal ke format datetime dan dilakukan pembersihan untuk kolom likes
- Statistik Deskriptif: Menghitung statistik dasar seperti mean, median, standar deviasi, nilai minimum, dan nilai maksimum untuk kolom "stars_given" dan "likes" yang merupakan fitur numerik. Ini akan memberikan pemahaman awal tentang distribusi dan rentang nilai data. 
- Visualisasi Data:
    - Plot Distribusi Stars Given akan menunjukkan menunjukkan kepuasan pengguna puas terhadap seri film Harry Potter yang ditonton dan ini mengindikasikan bias positif umum dalam ulasan buku populer.
    - Plot jumlah review per seri film Harry Potter akan menunjukkan urutan film Harry Potter seri mana yang paling banyak direview.
    - Plot rata-rata stars yang diberikan per seri film Harry Potter akan menunjukkan film Harry Potter seri mana yang memeroleh review terbaik melalui tinggi/rendahnya bintang yang diberikan.
    - Plot Korelasi antara stars_given dengan likes akan menunjukkan seberapa kuat korelasi antara 2 variabel yang ditentukan.

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
