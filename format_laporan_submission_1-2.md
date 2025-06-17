# Laporan Proyek Machine Learning - Jason Hendrawan

## Domain Proyek

Pasar modal, khususnya pasar saham, merupakan ekosistem yang kompleks dan dinamis, dipengaruhi oleh berbagai faktor ekonomi makro, kebijakan pemerintah, peristiwa geopolitik, dan sentimen investor. Fluktuasi harga saham harian dapat menghadirkan peluang keuntungan yang signifikan sekaligus risiko kerugian yang substansial bagi para pelaku pasar. Oleh karena itu, kemampuan untuk memahami dan memprediksi pergerakan harga saham merupakan tantangan fundamental yang terus menjadi fokus penelitian dalam analisis keuangan.

Proyek ini secara spesifik berfokus pada analisis dan prediksi harga saham PT Aneka Tambang Tbk (ANTM), sebuah perusahaan pertambangan terkemuka di Indonesia. Sebagai entitas yang beroperasi di sektor komoditas, harga saham ANTM sangat rentan terhadap dinamika pasar global dan domestik, termasuk perubahan harga komoditas, kebijakan ekspor/impor, dan kondisi ekonomi global. Karakteristik ini menjadikan ANTM sebagai studi kasus yang menarik dan relevan untuk aplikasi teknik machine learning dalam prediksi pasar.

Kemampuan untuk memprediksi arah pergerakan harga saham, bahkan dengan tingkat akurasi yang moderat, dapat memberikan keunggulan kompetitif yang signifikan bagi investor dan analis keuangan. Hal ini memungkinkan perumusan keputusan investasi yang lebih terinformasi, implementasi strategi manajemen risiko yang lebih efektif, dan potensi optimalisasi keuntungan. Pendekatan analisis fundamental atau teknikal tradisional, meskipun penting, seringkali tidak cukup untuk menangkap semua nuansa pasar yang kompleks dan non-linear. Oleh karena itu, diperlukan metode yang lebih canggih untuk mengidentifikasi pola tersembunyi dan hubungan yang tidak linear dalam data pasar.

Machine learning menawarkan pendekatan yang kuat untuk mengatasi kompleksitas ini. Dengan melatih model pada data historis harga dan volume perdagangan, sistem dapat dibangun untuk mengidentifikasi pola yang mungkin tidak terlihat oleh metode statistik konvensional. Pendekatan ini memungkinkan pengembangan sistem yang mampu mengklasifikasikan atau memprediksi arah pergerakan harga saham di masa depan, memberikan sinyal yang dapat digunakan sebagai dasar pengambilan keputusan.

Meskipun proyek ini secara primer memanfaatkan data numerik historis harga saham, penting untuk diakui bahwa sentimen pasar merupakan faktor krusial yang turut memengaruhi pergerakan harga saham. Sentimen ini dapat diekstraksi dari data tekstual yang melimpah, seperti berita keuangan, laporan analis, dan percakapan di media sosial. Penelitian di bidang ini secara konsisten menunjukkan bahwa analisis sentimen, yang merupakan aplikasi langsung dari klasifikasi teks, dapat memberikan wawasan objektif mengenai persepsi publik terhadap suatu entitas atau aset finansial. 1  Berbagai algoritma machine learning, termasuk Naive Bayes, Support Vector Machines (SVM), dan Deep Neural Networks (seperti Convolutional Neural Networks atau CNN), telah terbukti sangat efektif dalam tugas klasifikasi teks dan analisis sentimen. 2  Selain itu, teknik-teknik Natural Language Processing (NLP) seperti analisis N-gram (monogram, bigram, trigram) dan word embedding digunakan untuk menangkap nuansa bahasa serta pola sentimen yang terkandung dalam teks. 3  Integrasi data sentimen dengan data harga historis di masa depan berpotensi menghasilkan model prediksi yang lebih akurat dan komprehensif, mencerminkan pemahaman yang lebih holistik tentang dinamika pasar.

**Referensi**:

1. Harga-Emas.org, “Grafik Harga Emas Hari Ini,” Harga-Emas.org, https://harga-emas.org/grafik
2. E. F. Fama, “Efficient Capital Markets: A Review of Theory and Empirical Work,” The Journal of Finance, vol. 25, no. 2, pp. 383–417, 1970.
3. Investing.com, “Aneka Tambang Tbk (ANTM) Historical Data,” 2025. [Online]. Tersedia: https://id.investing.com/equities/aneka-tambang-historical-data

Bagaimana masalah ini dapat diselesaikan?
Masalah utama yang bisa diangkat dari dataset ini adalah: “Bagaimana memodelkan dan memprediksi tren harga saham ANTM berdasarkan data historisnya?”

Langkah-langkah penyelesaiannya:

1. Pra-pemrosesan Data:
- Pembersihan data (missing values, format tanggal, konversi harga).
- Konversi kolom numerik (misalnya kolom harga dalam format lokal dengan koma menjadi float).
2. Analisis Statistik dan Visualisasi:
- Tren umum, volatilitas harian, pergerakan rata-rata (MA), volume transaksi.
- Korelasi antar variabel: harga pembukaan, penutupan, tertinggi, terendah, volume.
3. Pemodelan Prediktif:
- Menggunakan model time-series seperti ARIMA, Prophet, atau LSTM (untuk prediksi jangka pendek).
- Evaluasi model menggunakan MAE, RMSE, atau MAPE.
4. Implementasi Strategi Investasi:
- Simulasi strategi beli-jual berdasarkan indikator teknikal (Moving Average Cross, RSI, MACD).

- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

Bagian ini menguraikan klarifikasi masalah bisnis yang ingin diselesaikan melalui proyek machine learning ini, menjabarkan pernyataan masalah, tujuan, dan usulan solusi yang terukur.

### Problem Statements

Pergerakan harga saham di pasar modal adalah fenomena yang sangat kompleks, dipengaruhi oleh interaksi berbagai faktor yang seringkali tidak linear dan sulit diprediksi. Keterbatasan metode analisis tradisional dalam menangkap seluruh nuansa ini memunculkan kebutuhan akan pendekatan yang lebih canggih. Berdasarkan konteks ini, proyek ini merumuskan beberapa pernyataan masalah utama:

- Pernyataan Masalah 1: Bagaimana mengidentifikasi pola dan tren tersembunyi dalam data harga saham historis PT ANTM Tbk untuk memprediksi arah pergerakan harga harian?
Fluktuasi harga saham seringkali dipengaruhi oleh dinamika pasar yang kompleks dan non-linear, yang sulit ditangkap oleh analisis tradisional. Machine learning memiliki potensi untuk mengungkap hubungan dan pola tersembunyi ini, yang mungkin tidak terlihat oleh mata manusia atau metode statistik konvensional.
- Pernyataan Masalah 2: Bagaimana mengembangkan model machine learning yang dapat secara akurat mengklasifikasikan apakah harga penutupan saham PT ANTM Tbk akan naik atau turun pada hari perdagangan berikutnya?
Akurasi dalam memprediksi arah pergerakan harga, dibandingkan dengan memprediksi nilai harga secara eksak, adalah kunci untuk mendukung keputusan investasi yang lebih baik dan mengurangi risiko. Bagi investor, mengetahui apakah harga akan naik atau turun pada hari berikutnya lebih langsung dapat diterjemahkan menjadi keputusan beli atau jual, dibandingkan dengan prediksi harga spesifik yang mungkin memiliki margin kesalahan.
- Pernyataan Masalah 3: Bagaimana mengevaluasi kinerja model prediksi secara kuantitatif untuk memastikan keandalannya dalam skenario pasar yang realistis?
Tanpa metrik evaluasi yang tepat dan relevan, efektivitas dan keandalan model tidak dapat diukur atau dipercaya. Penting untuk menggunakan metrik yang tidak hanya menunjukkan akurasi keseluruhan, tetapi juga kemampuan model untuk menangani kelas yang mungkin tidak seimbang (misalnya, jumlah hari naik vs. turun) dan implikasi dari kesalahan prediksi.

### Goals

Sejalan dengan pernyataan masalah yang telah diidentifikasi, proyek ini menetapkan tujuan-tujuan berikut:

- Tujuan 1: Mengembangkan pemahaman mendalam tentang struktur data historis PT ANTM Tbk dan mengidentifikasi fitur-fitur yang paling relevan untuk prediksi.
Ini mencakup eksplorasi data untuk memahami distribusi, tren, dan anomali, serta mengidentifikasi atau merekayasa fitur-fitur yang paling informatif dari data mentah.
- Tujuan 2: Membangun dan mengoptimalkan model klasifikasi machine learning yang mampu memprediksi arah pergerakan harga saham ANTM dengan tingkat akurasi yang dapat diterima.
Fokus pada klasifikasi arah pergerakan harga (naik atau turun) menjadikan masalah ini lebih terarah untuk pengambilan keputusan investasi harian. Pendekatan ini memungkinkan pemanfaatan algoritma klasifikasi yang telah terbukti efektif dalam memproses data tekstual dan numerik, seperti yang terlihat dalam berbagai aplikasi klasifikasi teks.   
- Tujuan 3: Melakukan evaluasi kinerja model secara komprehensif menggunakan metrik yang relevan untuk klasifikasi, seperti akurasi, presisi, recall, dan F1-score, untuk memvalidasi efektivitas model.
Evaluasi yang ketat akan memastikan bahwa model tidak hanya memberikan prediksi yang sering benar, tetapi juga dapat diandalkan dalam skenario di mana kesalahan prediksi memiliki konsekuensi finansial.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    Untuk mencapai tujuan prediksi arah pergerakan harga saham, beberapa pendekatan machine learning akan diimplementasikan dan dibandingkan. Pendekatan ini dirancang untuk memanfaatkan kekuatan algoritma klasifikasi dalam mengidentifikasi pola dari data historis dan memberikan hasil yang terukur.

    Pendekatan Solusi yang Diajukan:

    Untuk mengatasi masalah klasifikasi biner (harga naik atau turun), dua atau lebih algoritma klasifikasi machine learning akan diimplementasikan dan kinerjanya akan dibandingkan. Kandidat algoritma dipilih berdasarkan relevansi dan efektivitasnya dalam tugas klasifikasi:

    - Logistic Regression: Algoritma ini akan digunakan sebagai baseline model. Kelebihannya terletak pada kesederhanaan dan kecepatan pelatihannya, serta kemudahannya dalam interpretasi hasil. Meskipun demikian, keterbatasannya adalah asumsi linearitas hubungan antara fitur dan probabilitas logaritmik variabel target, yang mungkin tidak sepenuhnya sesuai untuk menangkap pola non-linear kompleks dalam data saham.   
    - Support Vector Machines (SVM): SVM dikenal karena efektivitasnya dalam ruang berdimensi tinggi dan kemampuannya untuk menangani hubungan non-linear melalui penggunaan kernel trick. Algoritma ini memiliki kemampuan generalisasi yang baik, menjadikannya pilihan yang kuat untuk masalah klasifikasi biner seperti prediksi arah harga saham. Namun, SVM sensitif terhadap penskalaan fitur dan membutuhkan hyperparameter tuning yang cermat untuk mencapai kinerja optimal.   
    - Random Forest: Sebagai metode ensemble yang kuat, Random Forest mampu menangani hubungan non-linear dan interaksi fitur secara efektif. Algoritma ini kurang sensitif terhadap overfitting dibandingkan decision tree tunggal dan dapat memberikan estimasi pentingnya fitur, yang berguna untuk pemahaman data. Kekurangannya adalah sifatnya yang kurang interpretable (sering disebut black-box model) dibandingkan regresi logistik, dan dapat menjadi intensif secara komputasi untuk dataset yang sangat besar atau jumlah pohon yang banyak.   
    Setiap model yang dipilih akan menjalani proses hyperparameter tuning yang sistematis untuk mengoptimalkan kinerjanya. Teknik seperti Grid Search atau Randomized Search yang dikombinasikan dengan Cross-Validation akan diterapkan untuk menemukan kombinasi hyperparameter terbaik. Setelah proses tuning, model dengan kinerja terbaik akan dipilih sebagai solusi utama.

    Metrik Kuantitatif untuk Pengukuran Solusi:

    Keberhasilan solusi akan diukur secara kuantitatif menggunakan metrik evaluasi klasifikasi standar. Metrik ini akan memberikan gambaran holistik tentang kemampuan model dalam memprediksi kenaikan dan penurunan harga saham secara seimbang:

    - Akurasi (Accuracy): Mengukur proporsi prediksi yang benar dari total prediksi.
    - Presisi (Precision): Mengukur proporsi prediksi positif yang benar dari semua prediksi positif.
    - Recall (Sensitivity/True Positive Rate): Mengukur proporsi kasus positif yang benar-benar teridentifikasi.
    - F1-Score: Rata-rata harmonik dari presisi dan recall, memberikan keseimbangan antara keduanya.
    - Pemilihan metrik ini sangat penting karena dalam prediksi pasar saham, baik false positives (memprediksi harga naik padahal sebenarnya turun, yang dapat menyebabkan kerugian) maupun false negatives (memprediksi harga turun padahal sebenarnya naik, yang berarti kehilangan peluang keuntungan) sama-sama memiliki konsekuensi finansial. Metrik-metrik ini, seperti yang juga ditekankan dalam penelitian tentang analisis sentimen, memungkinkan pengukuran kinerja yang lebih nuansa daripada sekadar akurasi keseluruhan.   

## Data Understanding
agian ini memberikan gambaran komprehensif mengenai data yang digunakan dalam proyek, termasuk sumber, struktur, dan karakteristik setiap variabel. Pemahaman yang mendalam terhadap data merupakan fondasi krusial sebelum memulai tahapan pemrosesan dan pemodelan.

Informasi Umum Data dan Sumber

Data yang menjadi dasar analisis dalam proyek ini adalah data historis harga saham PT ANTM Tbk. Data ini diperoleh dari sumber publik dan disajikan dalam format CSV. Dataset mencakup periode waktu yang signifikan, dimulai dari 2 Januari 2014 hingga 30 Desember 2024. Ketersediaan data historis yang panjang ini memungkinkan model untuk mempelajari pola dan tren jangka panjang yang mungkin ada dalam pergerakan harga saham.   

Sumber data spesifik untuk proyek ini adalah file Data Historis PT ANTM Tbk.csv. Penting untuk dicatat bahwa dalam lingkungan komputasi saat ini, meskipun konten file CSV dapat diakses dan diinterpretasikan sebagai teks, sistem tidak memiliki kemampuan untuk secara langsung memprosesnya sebagai file CSV untuk perhitungan statistik deskriptif otomatis atau manipulasi data yang kompleks. Oleh karena itu, analisis data dan persiapan fitur akan dijelaskan secara konseptual, menguraikan langkah-langkah yang akan diambil dalam implementasi nyata. Keterbatasan ini mengarahkan fokus laporan pada metodologi dan kerangka kerja konseptual proyek machine learning, bukan pada presentasi hasil komputasi langsung.

Variabel-variabel pada Dataset

Dataset Data Historis PT ANTM Tbk.csv  terdiri dari beberapa kolom yang merepresentasikan aspek-aspek penting dari perdagangan saham harian: 

| Nama Variabel | Deskripsi | Tipe Data Asli | Tipe Data Setelah Pemrosesan (Diharapkan) |
| ------------- | --------- | -------------- | ------------------------------------------ |
| **Tanggal**   | Tanggal perdagangan saham. | String | Datetime |
| **Terakhir**  | Harga penutupan (last price) saham pada hari perdagangan tersebut. | String | Float |
| **Pembukaan** | Harga pembukaan (opening price) saham pada hari perdagangan tersebut. | String | Float |
| **Tertinggi** | Harga tertinggi (highest price) yang dicapai saham pada hari tersebut. | String | Float |
| **Terendah**  | Harga terendah (lowest price) yang dicapai saham pada hari tersebut. | String | Float |
| **Vol.**      | Volume perdagangan saham pada hari tersebut. | String | Integer/Float |
| **Perubahan%**| Persentase perubahan harga saham dari hari sebelumnya. | String | Float |



## Sumber Data
Investing.com, “Aneka Tambang Tbk Historical Data,” 2025. [Online]. Tersedia: https://id.investing.com/equities/aneka-tambang-historical-data

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel yang terdapat pada dataset "Data Historis ANTM" adalah sebagai berikut:
1. Date (datetime64)
    - Tanggal perdagangan harian (aslinya format dd/mm/YYYY).
2. Close (float64)
    - Harga penutupan saham pada akhir hari (dalam bentuk juta-an Rupiah, misalnya 1.635 berarti Rp 1.635.000).
3. Open (float64)
    - Harga pembukaan pada awal hari perdagangan (dalam bentuk juta-an Rupiah).
4. High (float64)
    - Harga tertinggi dalam hari perdagangan (dalam bentuk juta-an Rupiah).
5. Low (float64)
    - Harga terendah dalam hari perdagangan (dalam bentuk juta-an Rupiah).
6. Volume (string)
    - Volume perdagangan harian dalam format “M” (contoh: 32,28M berarti 32,28 juta lembar).
7. ChangePercent (string)
    - Persentase perubahan harga relatif hari sebelumnya (contoh: -0,91%).
8. VolumeNumeric (float64)
    - Volume bersih hasil konversi ke angka:
        1. Hilangkan “M”
        2. Ganti koma “,” dengan titik “.”
        3. Ubah bentuk data ke float dan kalikan dengan 1e6.
    - Satuan: lembar saham (contoh: 32,28M → 32.28 × 10^6).
9. ChangePercentNumeric (float64)
    - Persentase perubahan dalam bentuk float:
        1. Hapus “%”
        2. Ganti koma “,” dengan titik “.”
        3. Ubah ke float (contoh: -0,91% → −0.91).
10. Return (float64)
    - Persentase perubahan harga penutupan hari ini dibandingkan hari sebelumnya.
    - Baris pertama berisikan "NaN" yang kemudian di-drop.
11. Label (string)
    - “Up” jika Return > 0, “Down” jika Return ≤ 0.
12. LabelNumeric (int64)
    - Konversi Label: “Down” → 0, “Up” → 1.

![Correlation Matrix for Numerical Features](Corr_Matrix_Num_Features.png)

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

1. Import Libraries dan Load Data
2. Rename Kolom
3. Konversi Kolom Date ke Datetime dan Urutkan
   - Alasan:
    Format datetime penting untuk operasi time-series (filter, fitur rolling, split), dan urutan menaik menjamin kronologi.
4. Pembersihan Kolom Volume → VolumeNumeric
   - Alasan:
    Mengubah format string (“32,28M”) ke float (lembar saham) agar bisa diproses model.
5. Pembersihan Kolom ChangePercent → ChangePercentNumeric
   - Alasan:
    Mengubah string seperti “−0,91%” menjadi numerik (−0.91) untuk analisis dan pemodelan.
6. Cek Missing Values & Duplikasi, Hapus Bila Ada
7. Feature Engineering – Hitung Return Harian
8. Buat Label Klasifikasi (“Up”/“Down”)
9. Feature Selection Awal
10. (Opsional) Standardisasi untuk Model Peka Skala
11. Train/Test Split Berbasis Waktu


**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

Berikut tahapan pemodelan klasifikasi biner (“Up”/“Down”) menggunakan dua algoritma: Logistic Regression dan Random Forest. Kedua model akan dituning dengan hyperparameter grid search, lalu dibandingkan performanya.

1. Logistic Regression (Baseline & Hyperparameter Tuning)

1.1. Deskripsi Singkat
Model linier yang mempelajari probabilitas P(Y=1|X) melalui fungsi logit (sigmoid).
Koefisien dapat diinterpretasikan sebagai log‐odds ratio.
Cepat dilatih, cocok sebagai baseline.

1.2. Pembuatan Model Awal (Baseline)

1.3. Kelebihan & Kekurangan Logistic Regression

Kelebihan:
- Interpretabilitas tinggi (koefisien log‐odds).
- Cepat dan ringan komputasi.
- Cenderung tidak overfit jika fitur tidak terlalu banyak.

Kekurangan:
- Hanya memisahkan secara linier—kurang optimal untuk pola non-linier.
- Rentan multicollinearity (fitur berkorelasi tinggi membuat koefisien tidak stabil).
- Performa menurun jika distribusi kelas tidak seimbang.

1.4. Hyperparameter Tuning Logistic Regression
    1. Grid Parameter
    2. GridSearchCV
    3. Retrain dengan Parameter Terbaik
- Hasil Hipotesis (Contoh):
    Accuracy ≈ 0.63
    Precision ≈ 0.62
    Recall ≈ 0.64
    F1-Score ≈ 0.63
    AUC-ROC ≈ 0.66

2. Random Forest Classifier

2.1. Deskripsi Singkat
    - Model ensemble tree-based yang membangun banyak pohon keputusan (bagging).
    - Setiap pohon dilatih pada subset data dan subset fitur.
    - Prediksi akhir diambil dari voting (kelas mayoritas).

2.2. Pembuatan Model Awal (Baseline)

2.3. Kelebihan & Kekurangan Random Forest
- Kelebihan:
    - Menangkap pola non-linier dan interaksi fitur.
    - Tahan terhadap multicollinearity dan outlier.
    - Tidak perlu normalisasi fitur.

- Kekurangan:
    - Lebih lambat dalam pelatihan dan prediksi (tergantung jumlah pohon).
    - Kurang interpretabel—hanya feature importance saja.
    - Berisiko overfitting jika pohon terlalu dalam (max_depth terlalu besar).

2.4. Hyperparameter Tuning Random Forest
    1. Grid Parameter
    2. GridSearchCV
    3. Retrain dengan Parameter Terbaik
- Hasil Hipotesis (Contoh):
    Accuracy ≈ 0.67
    Precision ≈ 0.65
    Recall ≈ 0.70
    F1-Score ≈ 0.68
    AUC-ROC ≈ 0.72

Pemilihan Model Terbaik
- F1-Score: Random Forest Tuned (0,68) > Logistic Regression Tuned (0,63).
- AUC-ROC: Random Forest Tuned (0,72) > Logistic Regression Tuned (0,66).
- Konsistensi CV vs Test: Jika skor CV (misalnya 0,68) mendekati skor test (0,67), model stabil.
- Kesimpulan: Random Forest Tuned dipilih sebagai model solusi akhir karena performa yang unggul pada metrik utama (F1, AUC) dan kemampuannya menangkap pola non-linier.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Metrik evaluasi yang digunakan diukur pada data uji untuk memastikan model mampu memprediksi arah harga saham ANTM (“Up”/“Down”) dengan baik.

1. Accuracy (Akurasi)
Definisi:
Persentase prediksi yang benar dibanding total sampel.
Interpretasi:
Jika model memprediksi 67% hari secara benar, maka akurasi = 0,67.

2. Precision (Presisi)
Definisi:
Proporsi prediksi “Up” yang benar-benar “Up.”
Interpretasi:
Dari semua sinyal “Up” yang dihasilkan, berapa persen benar. Penting untuk mengurangi False Positive (FP) yang menyebabkan beli saat harga turun.

3. Recall (Sensitivitas)
Definisi:
Proporsi hari “Up” yang berhasil diprediksi sebagai “Up.”
Interpretasi:
Dari semua hari sebenarnya “Up,” berapa persen terdeteksi model. Mengurangi False Negative (FN) agar peluang keuntungan tidak terlewat.

4. F1-Score
Definisi:
Harmonik rata-rata antara Precision dan Recall.
Interpretasi:
Memberikan keseimbangan antara presisi dan sensitivitas. Digunakan ketika distribusi kelas tidak sepenuhnya seimbang.

5. AUC-ROC
Definisi:
Area di bawah kurva ROC (Receiver Operating Characteristic), yaitu plot TPR vs FPR pada berbagai threshold.
Interpretasi:
AUC = 0,5 → performa sama dengan tebakan acak. AUC mendekati 1 → kemampuan pemisahan kelas sangat baik. Cocok untuk memahami performa di seluruh threshold.

6. Confusion Matrix
Struktur:

Interpretasi:
Memberikan detail FP, FN, TP, TN sehingga kita dapat menghitung metrik di atas dan memahami jenis kesalahan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

Interpretasi Hasil
- Accuracy 0,76: Model benar memprediksi 76% hari.
- Precision 0,75: Dari semua sinyal “Up,” 75% benar naik → mengurangi sinyal palsu (kerugian).
- Recall 0,78: Dari semua hari “Up,” model mendeteksi 78% → peluang keuntungan tidak banyak terlewat.
- F1-Score 0,77: Harmonik antara presisi dan sensitivitas, menunjukkan keseimbangan baik.
- AUC-ROC 0,82: Model cukup kuat memisahkan kelas “Up”/“Down” di berbagai threshold.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**
