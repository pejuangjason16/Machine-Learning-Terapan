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

- Tujuan 1: Melakukan eksplorasi dan pemahaman lebih dalam mengenai dataset yang digunakan agar dapat mengetahui film Harry Potter manakah yang paling disukai/direkomendasikan berdasarkan dataset yang digunakan.
- Tujuan 2: Membangun model-model sistem rekomendasi terbaik berdasarkan Content-Based Filtering dan Collaborative Filtering untuk menghasilkan rekomendasi film Harry Potter yang terbaik yang dibuat sampai sekarang.
- Tujuan 3: Melakukan evaluasi kinerja pada model-model yang dibangun untuk melihat hasil evaluasi dan performa dari sistem rekomendasi yang dibangun memang menentukan kualitas dari rekomendasi yang diberikan.

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

**Untuk Content-Based Filtering**

- Mengubah teks deskripsi `description` film menjadi matriks representasi numerik menggunakan TF-IDF Vectorizer sehingga dapat mengidentifikasi kata-kata kunci yang penting dari setiap deskripsi.

**Untuk Collaborative Filtering**

- Melakukan Data Loading dengan menggunakan library surprise untuk memuat Data `rating`.
- Melakukan split Train-Test dengan membagi Dataset menjadi data latih dan data uji untuk mengevaluasi performa model.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
from collections import defaultdict
from google.colab import drive
drive.mount('/content/drive')

# Load dataset
path = '/content/drive/MyDrive/Film Harry Potter.csv'
df = pd.read_csv(path)

# Hapus kolom tidak relevan
df.drop(columns=['Unnamed: 0'], inplace=True)

# Bersihkan kolom 'likes'
df['likes'] = df['likes'].str.replace(' likes', '', regex=False)
df['likes'] = df['likes'].str.replace(',', '', regex=False).astype(int)

# Konversi tanggal dengan format eksplisit
df['date'] = pd.to_datetime(df['date'], format='%b%d,%Y', errors='coerce')

# Cek missing values
print(df.isnull().sum())

# Drop baris dengan missing pada kolom penting
df = df.dropna(subset=['description', 'stars_given'])

# TF-IDF dari kolom description
tfidf_matrix = vectorizer.fit_transform(df['description'])

# Melakukan Data Loading menggunakan library surprise untuk memuat Data `rating`
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['name', 'book', 'stars_given']], reader)

# Menyimpan split untuk evaluasi
trainset_eval, testset_eval = train_test_split(data, test_size=0.2, random_state=42)
```

## Modeling

### Content-Based Filtering dari Deskripsi

Dalam proyek ini, dua pendekatan sistem rekomendasi diimplementasikan: Content-Based Filtering dan Collaborative Filtering. Masing-masing pendekatan memiliki kelebihan dan kekurangan yang menjadi pertimbangan dalam pemilihan solusi.

***Content-Based Filtering***

Pendekatan ini merekomendasikan item berdasarkan kemiripan atribut atau konten dari item tersebut. Dalam kasus ini, atribut yang digunakan adalah deskripsi dari setiap film.

Model yang Digunakan:

*Cosine Similarity:* Setelah deskripsi diubah menjadi vektor numerik, cosine similarity digunakan untuk mengukur kemiripan antara dua film. Semakin tinggi nilai cosine similarity, semakin mirip kedua film tersebut.

Berikut adalah hasil dari sistem rekomendasi content-based filtering:

| Index | Book                                      | Name    | Stars | Deskripsi                                             |
|-------|-------------------------------------------|---------|--------|--------------------------------------------------------|
| 362   | Harry Potter and the Chamber of Secrets   | Jayson  | 4.0    | (A-) 83% \| Very GoodNotes: A bit bland at times...    |
| 8     | Harry Potter and the Sorcerer's Stone     | Jayson  | 4.0    | (A-) 83% \| Very GoodNotes: Despite a weak clim...     |
| 365   | Harry Potter and the Chamber of Secrets   | Jayson  | 4.0    | (A-) 83% \| Very GoodNotes: A clever conjuration...    |

---

**Kelebihan dan Kekurangan Content-Based Filtering**

**Kelebihan:**
- Independensi Pengguna: Model tidak memerlukan data dari pengguna lain untuk memberikan rekomendasi. Rekomendasi untuk satu pengguna tidak dipengaruhi oleh pengguna lain, sehingga dapat mengatasi masalah untuk pengguna baru (user cold start).
- Transparansi: Rekomendasi yang diberikan mudah dijelaskan. Kita dapat mengatakan, "Film B direkomendasikan karena memiliki deskripsi cerita yang mirip dengan Film A yang Anda sukai," yang dapat meningkatkan kepercayaan pengguna.
- Tidak Ada Masalah untuk Item Baru: Selama sebuah film baru memiliki deskripsi, ia bisa langsung masuk ke dalam sistem rekomendasi tanpa perlu menunggu data rating dari pengguna.

**Kekurangan:**
- Keterbatasan Fitur: Kualitas rekomendasi sangat bergantung pada data deskripsi yang tersedia. Jika deskripsi kurang detail atau tidak representatif, maka rekomendasinya akan kurang relevan.
- Minim Kejutan (Low Serendipity): Model ini cenderung merekomendasikan item yang sangat mirip dengan apa yang sudah disukai pengguna. Hal ini membatasi pengguna untuk menemukan kategori atau genre baru yang mungkin juga mereka sukai, menciptakan "gelembung filter" (filter bubble).

---

***Collaborative Filtering***

Pendekatan ini merekomendasikan item berdasarkan preferensi dari pengguna lain yang memiliki selera serupa. Dalam proyek ini, data rating (stars_given) dari pengguna (name) terhadap film (book) digunakan untuk membangun model.

Model yang Digunakan:

*SVD (Singular Value Decomposition):* Sebuah algoritma faktorisasi matriks yang populer digunakan dalam sistem rekomendasi. SVD akan menguraikan matriks interaksi pengguna-item menjadi beberapa matriks faktor yang lebih kecil, yang kemudian digunakan untuk memprediksi rating yang belum diberikan oleh pengguna.

Berikut adalah hasil dari sistem rekomendasi collaborative filtering:

**User: MirandaReads**  
- Harry Potter and the Cursed Child: Parts One and Two (prediksi: 3.10)

**User: Lora**  
- Harry Potter and the Goblet of Fire (prediksi: 4.77)  
- Harry Potter and the Half-Blood Prince (prediksi: 4.73)  
- Harry Potter and the Deathly Hallows (prediksi: 4.72)

**User: ★Jess**  
- Harry Potter and the Goblet of Fire (prediksi: 5.00)  
- Harry Potter and the Prisoner of Azkaban (prediksi: 4.74)  
- Harry Potter and the Half-Blood Prince (prediksi: 4.73)

**User: Matthew** 
- Harry Potter and the Goblet of Fire (prediksi: 4.96)  
- Harry Potter and the Half-Blood Prince (prediksi: 4.84)  
- Harry Potter and the Order of the Phoenix (prediksi: 4.65)

**User: Khanh,firstofhername,motherofbunnies** 
- Harry Potter and the Half-Blood Prince (prediksi: 4.86)  
- Harry Potter and the Goblet of Fire (prediksi: 4.84)  
- Harry Potter and the Deathly Hallows (prediksi: 4.77)

---

**Kelebihan dan Kekurangan Collaborative Filtering**

**Kelebihan:**

- Mampu Memberikan Rekomendasi Lintas Genre: Model ini tidak bergantung pada konten item. Ia dapat menemukan hubungan yang tak terduga antar item (misalnya, penggemar Film A juga menyukai Film Z, meskipun genre keduanya berbeda) dan memberikan rekomendasi yang mengejutkan namun relevan (high serendipity).
- Tidak Memerlukan Analisis Konten: Model bekerja murni berdasarkan interaksi pengguna (rating), sehingga tidak memerlukan proses rekayasa fitur yang rumit pada deskripsi atau atribut item.

**Kekurangan:**

- Masalah Cold Start: Ini adalah kelemahan utama. Model kesulitan memberikan rekomendasi untuk pengguna baru yang belum memiliki riwayat rating, atau merekomendasikan item baru yang belum pernah diberi rating oleh siapapun.
- Ketersebaran Data (Sparsity): Pada dataset yang besar, matriks interaksi pengguna-item seringkali sangat kosong (setiap pengguna hanya memberi rating pada sebagian kecil item). Hal ini dapat menyulitkan pencarian "tetangga" yang relevan dan dapat menurunkan kualitas rekomendasi.
- Kurang Transparan: Lebih sulit untuk menjelaskan mengapa sebuah item direkomendasikan, karena hanya didasarkan pada selera pengguna lain yang dianggap "mirip" secara matematis.

---

## Evaluation

Evaluasi dilakukan pada model Collaborative Filtering untuk mengukur seberapa baik model dapat memprediksi rating yang akan diberikan oleh pengguna.

**Metrik Evaluasi**

Metrik evaluasi yang digunakan untuk model Content-Based Filtering adalah Precision@K dan Recall@K yang mengukur performa sistem dalam menyarankan item relevan di antara Top-K hasil rekomendasi. Berikut adalah rumusnya:

**Precision@K**:

$$
\text{Precision@K} = \frac{\text{Jumlah item relevan di Top-K}}{K}
$$

**Recall@K**:

$$
\text{Recall@K} = \frac{\text{Jumlah item relevan di Top-K}}{\text{Total item relevan}}
$$

Penjelasan:
- Precision@K mengukur **akurasi**: seberapa banyak rekomendasi yang benar dari total yang disarankan.
- Recall@K mengukur **cakupan**: seberapa besar porsi item relevan berhasil ditangkap oleh sistem.
- Nilai berkisar antara 0 dan 1, makin tinggi makin baik.

Metrik evaluasi yang digunakan untuk model Collaborative Filtering adalah Root Mean Squared Error (RMSE). RMSE mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual. Semakin kecil nilai RMSE, semakin baik performa model dalam memprediksi rating. Berikut adalah rumusnya:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{r}_i - r_i)^2}
$$

Penjelasan:
- ri_hat: rating yang diprediksi oleh model untuk item ke-i.
- ri: rating aktual yang diberikan pengguna untuk item ke-i.
- n : jumlah total data prediksi yang dievaluasi.
- RMSE menunjukkan rata-rata jarak (dalam kuadrat) antara prediksi dan kenyataan.
- Nilai RMSE semakin kecil → prediksi model semakin akurat.

**Evaluasi Content-Based Filtering (CBF)**
- "Precision @3: 0.2486", artinya dari 3 rekomendasi teratas yang diberikan untuk setiap review, rata-rata sekitar 24.86% benar-benar dianggap relevan (yaitu berasal dari review lain oleh user yang sama).
  - Sekitar 1 dari 4 rekomendasi masuk ke kategori yang dianggap relevan.
  - Ini menunjukkan bahwa model CBF memiliki kemampuan cukup baik dalam memilih konten yang serupa dan relevan bagi pengguna.

- "Recall@3: 0.1690", artinya Dari seluruh review yang seharusnya relevan (review lain milik user tersebut), model berhasil merekomendasikan 16.9% di antaranya.
  - Model tidak mampu menjangkau semua item relevan, tapi masih berhasil meng-cover sebagian.
  - Ini merupakan hal yang normal dalam skenario dengan jumlah review relevan terbatas dan hanya mengambil Top-3 rekomendasi.

**Evaluasi Collaborative Filtering (SVD)**
- RMSE (Root Mean Squared Error): 0.8544
- Dengan skala rating 1–5, nilai 0.8544 menunjukkan bahwa rata-rata prediksi model hanya meleset sekitar 0.85 poin.
- Ini bisa terbilang cukup akurat untuk dataset kecil seperti ulasan film Harry Potter.
- Nilai ini menandakan bahwa model SVD cukup andal dalam memahami pola rating pengguna dan bisa digunakan untuk rekomendasi yang layak.

Model-model yang telah dibangun untuk proyek kali ini beserta hasil evaluasinya mampu untuk menjawab semua permasalahan yang ingin diselesaikan pada proyek ini. Model *Singular Value Decomposition (SVD)* yang dibangun melalui pendekatan Collaborative Filtering mampu memberikan rekomendasi yang akurat berdasarkan prediksi rating pengguna, serta model SVD merupakan model yang lebih akurat dan unggul dalam performa. Sedangkan Model *Cosine Similarity* yang dibangun melalui pendekatan Content-Based Filtering mampu menyarankan item berdasarkan kemiripan isi/konten review, atau tidak punya banyak data rating. Meskipun model SVD lebih akurat dan unggul dalam performa, namun model *Cosine Similarity* tetap relevan dan berguna walau cakupannya terbatas. Sehingga model SVD mampu merekomendasikan film Harry Potter seri manakah yang paling disukai berdasarkan dataset ulasan Film Harry Potter yang digunakan. Setiap pengguna mampu memberikan rekomendasi yang akurat, seperti pengguna Lora yang merekomendasikan film Harry Potter and the Goblet of Fire (dengan prediksi: 4.77), film Harry Potter and the Half-Blood Prince (dengan prediksi: 4.73), film Harry Potter and the Deathly Hallows (dengan prediksi: 4.72). Pengguna ★Jess yang merekomendasikan film Harry Potter and the Goblet of Fire (dengan prediksi: 5.00), film Harry Potter and the Prisoner of Azkaban (dengan prediksi: 4.74), film Harry Potter and the Half-Blood Prince (dengan prediksi: 4.73). dan beberapa pengguna lainnya. Dari model yang dibangun melalui pendekatan Content-Based Filtering, beberapa film Harry Potter yang direkomendasikan berdasarkan kemiripan isi/konten review seperti *Harry Potter and the Chamber of Secrets* dengan keterangan deskripsinya [(A-) 83%] (Very GoodNotes: A bit bland at times..), *Harry Potter and the Sorcerer's Stone* dengan keterangan deskripsinya [(A-) 83%] (Very GoodNotes: Despite a weak clim...), *Harry Potter and the Chamber of Secrets* dengan keterangan deskripsinya [(A-) 83%] (Very GoodNotes: A clever conjuration...) sama-sama berasal dari user Jayson dengan bintang rating yang diberikan sebesar 4.0. Hasil evaluasi yang sudah dijabarkan sebelumnya juga mendukung pernyataan bahwa model *Singular Value Decomposition (SVD)* yang dibangun melalui pendekatan Collaborative Filtering mampu memberikan rekomendasi yang akurat karena cukup unggul dalam hal performa melalui nilai RMSE yang dihasilkan sebesar 0.8544.

Dengan demikian, solusi yang diajukan pada waktu awal yaitu model yang dibangun melalui pendekatan *Content-Based Filtering* dengan memanfaatkan deskripsi atau sinopsis dari setiap film yang merekomendasikan film berdasarkan kemiripan kontennya dan menggunakan teknik  TF-IDF Vectorizer dan Cosine Similarity belum memiliki dampak yang cukup besar karena hasilnya masih belum cukup akurat jika dibandingkan dengan model yang dibangun melalui *Collaborative Filtering* dengan memanfaatkan data rating yang diberikan oleh pengguna yang merekomendasikan film berdasarkan preferensi dari pengguna lain yang memiliki selera serupa, dengan menggunakan algoritma Singular Value Decomposition (SVD), serta metrik evaluasi yang digunakan adalah Root Mean Squared Error (RMSE) untuk mengukur tingkat kesalahan prediksi rating dari model yang memiliki dampak yang cukup besar karena hasil evaluasi dan hasil rekomendasinya. Sehingga semua permasalahan yang ingin diselesaikan dan tujuan yang ingin dicapai, semuanya berhasil dijawab pada proyek kali ini.

**---Ini adalah bagian akhir laporan---**
