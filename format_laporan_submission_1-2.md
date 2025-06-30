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
- Melihat tren umum, volatilitas harian, pergerakan rata-rata (MA), volume transaksi.
- Melihat korelasi antar variabel: harga pembukaan, penutupan, tertinggi, terendah, volume.
3. Pemodelan Prediktif:
- Menggunakan model time-series seperti ARIMA (untuk prediksi jangka pendek).
- Evaluasi model menggunakan MAE, RMSE, atau MAPE.
4. Implementasi Strategi Investasi:
- Simulasi strategi beli-jual berdasarkan indikator teknikal (Moving Average Cross, RSI, MACD).

- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

Bagian ini menguraikan klarifikasi masalah bisnis yang ingin diselesaikan melalui proyek machine learning ini, menjabarkan pernyataan masalah, tujuan, dan usulan solusi yang terukur.

Bagian laporan ini mencakup:

### Problem Statements

Pergerakan harga saham di pasar modal adalah fenomena yang sangat kompleks, dipengaruhi oleh interaksi berbagai faktor yang seringkali tidak linear dan sulit diprediksi. Keterbatasan metode analisis tradisional dalam menangkap seluruh nuansa ini memunculkan kebutuhan akan pendekatan yang lebih canggih. Berdasarkan konteks ini, proyek ini merumuskan beberapa pernyataan masalah utama:

- Pernyataan Masalah 1: Bagaimana mengidentifikasi pola dan tren tersembunyi dalam data harga saham historis PT ANTM Tbk untuk memprediksi arah pergerakan harga harian?
    - Fluktuasi harga saham seringkali dipengaruhi oleh dinamika pasar yang kompleks dan non-linear, yang sulit ditangkap oleh analisis tradisional. Machine learning memiliki potensi untuk mengungkap hubungan dan pola tersembunyi ini, yang mungkin tidak terlihat oleh mata manusia atau metode statistik konvensional.
- Pernyataan Masalah 2: Bagaimana mengembangkan model machine learning yang dapat secara akurat mengklasifikasikan apakah harga penutupan saham PT ANTM Tbk akan naik atau turun pada hari perdagangan berikutnya?
    - Akurasi dalam memprediksi arah pergerakan harga, dibandingkan dengan memprediksi nilai harga secara eksak, adalah kunci untuk mendukung keputusan investasi yang lebih baik dan mengurangi risiko. Bagi investor, mengetahui apakah harga akan naik atau turun pada hari berikutnya lebih langsung dapat diterjemahkan menjadi keputusan beli atau jual, dibandingkan dengan prediksi harga spesifik yang mungkin memiliki margin kesalahan.
- Pernyataan Masalah 3: Bagaimana mengevaluasi kinerja model prediksi secara kuantitatif untuk memastikan keandalannya dalam skenario pasar yang realistis?
    - Tanpa metrik evaluasi yang tepat dan relevan, efektivitas dan keandalan model tidak dapat diukur atau dipercaya. Penting untuk menggunakan metrik yang tidak hanya menunjukkan akurasi keseluruhan, tetapi juga kemampuan model untuk menangani kelas yang mungkin tidak seimbang (misalnya, jumlah hari naik vs. turun) dan implikasi dari kesalahan prediksi.

### Goals

Sejalan dengan pernyataan masalah yang telah diidentifikasi, proyek ini menetapkan tujuan-tujuan berikut:

- Tujuan 1: Mengembangkan pemahaman mendalam tentang struktur data historis PT ANTM Tbk dan mengidentifikasi fitur-fitur yang paling relevan untuk prediksi.
    - Hal ini mencakup eksplorasi data untuk memahami distribusi, tren, dan anomali, serta mengidentifikasi atau merekayasa fitur-fitur yang paling informatif dari data mentah.
- Tujuan 2: Membangun dan mengoptimalkan model klasifikasi machine learning yang mampu memprediksi arah pergerakan harga saham ANTM dengan tingkat akurasi yang dapat diterima.
    Fokus pada klasifikasi arah pergerakan harga (naik atau turun) menjadikan masalah ini lebih terarah untuk pengambilan keputusan investasi harian. Pendekatan ini memungkinkan pemanfaatan algoritma klasifikasi yang telah terbukti efektif dalam memproses data tekstual dan numerik, seperti yang terlihat dalam berbagai aplikasi klasifikasi teks.   
- Tujuan 3: Melakukan evaluasi kinerja model secara komprehensif menggunakan metrik yang relevan untuk klasifikasi, seperti akurasi, presisi, recall, dan F1-score, untuk memvalidasi efektivitas model.
    - Evaluasi yang ketat akan memastikan bahwa model tidak hanya memberikan prediksi yang sering benar, tetapi juga dapat diandalkan dalam skenario di mana kesalahan prediksi memiliki konsekuensi finansial.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

### Solution statements
Untuk mencapai tujuan prediksi arah pergerakan harga saham, beberapa pendekatan machine learning akan digunakan untuk diimplementasikan dan dibandingkan. 

**Pendekatan Solusi yang Diajukan**:

Akan digunakan dua atau lebih algoritma klasifikasi machine learning untuk diimplementasikan dan kinerjanya akan dibandingkan, guna untuk mengatasi masalah klasifikasi biner (harga naik atau turun). Kandidat algoritma dipilih berdasarkan relevansi dan efektivitasnya dalam tugas klasifikasi:

- Logistic Regression: Algoritma ini akan digunakan sebagai baseline model. Kelebihan dari algoritma ini terletak pada kesederhanaan dan kecepatan pelatihannya, juga mampu untuk menginterpretasikan hasil dengan mudah. Meskipun begitu, terdapat keterbatasan pada model ini dimana asumsi linearitas hubungan antara fitur dan probabilitas logaritmik variabel target mungkin tidak sepenuhnya sesuai untuk menangkap pola non-linear kompleks dalam data saham.   
- Support Vector Machines (SVM): SVM memiliki tingkat efektivitas yang cukup baik dalam ruang berdimensi tinggi dan memiliki kemampuan yang cukup mumpuni untuk menangani hubungan non-linear melalui penggunaan kernel trick. Model ini mampu melakukan generalisasi dengan cukup baik, sehingga bisa menjadi salah satu pilihan yterbaik untuk masalah klasifikasi biner seperti prediksi arah harga saham. Perlu diperhatikan SVM sensitif terhadap penskalaan fitur dan membutuhkan hyperparameter tuning yang cermat untuk mencapai kinerjanya yang paling optimal.   
- Random Forest: Sebagai metode ensemble yang kuat, Random Forest mampu menangani hubungan non-linear dan interaksi fitur secara efektif. Perlu diperhatikan bahwa algoritma ini kurang sensitif terhadap overfitting dibandingkan decision tree tunggal dan juga mampu memberikan estimasi pentingnya fitur, yang berguna untuk memperdalam pemahaman data. Kekurangannya lebih sulit untuk diinterpretasikan dibandingkan regresi logistik, dan dapat menjadi intensif secara komputasi untuk dataset yang sangat besar atau jumlah pohon yang banyak.

    Setiap model yang dipilih akan menjalani proses hyperparameter tuning yang sistematis untuk mengoptimalkan kinerjanya. Teknik seperti Grid Search atau Randomized Search yang dikombinasikan dengan Cross-Validation akan diterapkan untuk menemukan kombinasi hyperparameter terbaik. Setelah proses tuning, model dengan kinerja terbaik akan dipilih sebagai solusi utama.

**Metrik Kuantitatif untuk Pengukuran Solusi**:

Keberhasilan solusi akan dilihat melalui pengukuran yang sudah dilakukan secara kuantitatif menggunakan metrik evaluasi klasifikasi standar. Nilai - nilai yang diperoleh dari metrik ini akan memberikan gambaran mengenai kemampuan model dalam memprediksi kenaikan dan penurunan harga saham secara seimbang:

- Akurasi (Accuracy): Mengukur proporsi prediksi yang benar dari total prediksi.
- Presisi (Precision): Mengukur proporsi prediksi positif yang benar dari semua prediksi positif.
- Recall (Sensitivity/True Positive Rate): Mengukur proporsi kasus positif yang benar-benar teridentifikasi.
- F1-Score: Rata-rata harmonik dari presisi dan recall, memberikan keseimbangan antara keduanya.
  
    Pemilihan metrik ini sangat penting karena dalam prediksi pasar saham, baik false positives (memprediksi harga naik padahal sebenarnya turun, yang dapat menyebabkan kerugian) maupun false negatives (memprediksi harga turun padahal sebenarnya naik, yang berarti kehilangan peluang keuntungan) sama-sama memiliki konsekuensi finansial. Metrik-metrik ini, seperti yang juga ditekankan dalam penelitian tentang analisis sentimen, memungkinkan pengukuran kinerja yang lebih nuansa daripada sekadar akurasi keseluruhan.   

## Data Understanding
Bagian ini akan memberikan gambaran secara menyeluruh mengenai data yang digunakan dalam proyek, termasuk sumber, struktur, dan karakteristik setiap variabel. Pemahaman yang mendalam terhadap data merupakan fondasi krusial sebelum memulai tahapan pemrosesan dan pemodelan.

**Informasi Umum Data dan Sumber**:

Data yang menjadi dasar analisis dalam proyek ini adalah data historis harga saham PT ANTM Tbk. Data ini diperoleh dari sumber publik dan disajikan dalam format CSV. Dataset mencakup periode waktu yang signifikan, dimulai dari 2 Januari 2014 hingga 30 Desember 2024. Ketersediaan data historis yang panjang ini memungkinkan model untuk mempelajari pola dan tren jangka panjang yang mungkin ada dalam pergerakan harga saham.   

### Variabel-variabel pada Dataset

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

*** Tabel 1: Variabel-variabel pada Dataset Historis PT ANTM Tbk

### Eksplorasi Data dan Analisis Statistik Deskriptif

Tahap eksplorasi data (EDA) dan analisis statistik deskriptif sangat penting untuk memahami karakteristik data, mengidentifikasi potensi masalah kualitas data, dan merumuskan strategi rekayasa fitur yang efektif. Meskipun tidak dapat dilakukan secara langsung dalam lingkungan ini, langkah-langkah konseptualnya adalah sebagai berikut:

- Pemuatan Data: Data akan dimuat ke dalam struktur data yang sesuai, seperti DataFrame Pandas, untuk memfasilitasi manipulasi dan analisis yang efisien.
- Pembersihan Data Awal: Proses ini melibatkan penanganan nilai yang hilang (jika ada), yang dapat dilakukan dengan imputasi (misalnya, forward-fill untuk data deret waktu yang berurutan) atau penghapusan baris/kolom jika proporsi nilai hilang sangat kecil. Selain itu, konversi tipe data kolom Tanggal ke format datetime dan kolom Vol. serta Perubahan% dari string (dengan sufiks 'M', 'B', '%') menjadi numerik (float) adalah langkah esensial. Tipe data yang benar sangat krusial untuk memungkinkan operasi matematika dan analisis deret waktu yang akurat. Data volume dan persentase perubahan dalam format string tidak dapat langsung digunakan dalam model numerik.
- Statistik Deskriptif: Menghitung statistik dasar seperti mean, median, standar deviasi, nilai minimum, dan nilai maksimum untuk kolom harga (Terakhir, Pembukaan, Tertinggi, Terendah) dan volume (Vol.) akan memberikan pemahaman awal tentang distribusi dan rentang nilai data. Ini membantu dalam mengidentifikasi anomali atau pola umum.
- Visualisasi Data:
    - Plot deret waktu harga penutupan (Terakhir) akan digunakan untuk mengamati tren jangka panjang, pola musiman, atau anomali signifikan dalam pergerakan harga.
    - Histogram distribusi Perubahan% akan memberikan wawasan tentang volatilitas harian saham, menunjukkan seberapa sering terjadi kenaikan atau penurunan harga dalam rentang tertentu.
    - Scatter plot antara Vol. dan Perubahan% dapat membantu mencari korelasi antara volume perdagangan dan perubahan harga.
- Pembentukan Variabel Target: Untuk masalah klasifikasi prediksi arah pergerakan harga, variabel target biner (Arah_Gerak) akan dibuat dari kolom Perubahan%. Variabel ini akan diberi nilai 1 jika Perubahan% > 0 (mengindikasikan harga naik) dan 0 jika Perubahan% <= 0 (mengindikasikan harga turun atau stagnan). Transformasi ini mengubah masalah regresi (memprediksi nilai harga) menjadi masalah klasifikasi (memprediksi arah), yang seringkali lebih praktis untuk pengambilan keputusan investasi harian dan lebih sesuai dengan fokus klasifikasi yang relevan dengan penelitian yang tersedia.
- Identifikasi Outlier: Menggunakan metode statistik atau visualisasi, outlier dalam data (misalnya, lonjakan harga atau volume yang ekstrem) akan dideteksi. Outlier ini perlu ditangani dengan hati-hati karena dapat memengaruhi kinerja model secara negatif jika tidak ditangani dengan tepat.

Pada tahap pemahaman data ini merupakan tahapan yang penting seblum melangkah ke tahapan selanjutnya, karena tahapan ini akan memastikan bahwa data siap untuk diproses dan diinterpretasikan dengan benar dalam konteks pemodelan machine learning.

## Data Preparation
Tahap persiapan data merupakan fase yang krusial peranannya dalam siklus hidup proyek machine learning, di mana data mentah diubah menjadi format yang sesuai untuk pemodelan dan memiliki tujuan untuk memaksimalkan kinerja dan keandalan model.

### Teknik Data Preparation yang Diterapkan
Proses persiapan data akan melibatkan serangkaian langkah yang terurut dan:
- Konversi Tipe Data:
    - Kolom Tanggal akan dikonversi dari format string ke objek datetime. Ini sangat penting untuk memungkinkan operasi deret waktu, seperti pengurutan kronologis dan pembuatan fitur berbasis waktu.
    - Kolom Vol. dan Perubahan% yang saat ini dalam format string (dengan sufiks seperti 'M' untuk juta, 'B' untuk miliar, dan '%' untuk persentase) akan dikonversi menjadi tipe data numerik (float).
    - Justifikasi: Tipe data yang benar adalah prasyarat fundamental untuk melakukan operasi matematika dan analisis statistik. Data volume dan persentase perubahan yang masih dalam format string tidak dapat diproses secara numerik oleh algoritma machine learning, sehingga konversi ini mutlak diperlukan untuk mengintegrasikannya ke dalam model.
- Penanganan Nilai Hilang (Missing Values):
    - Meskipun data yang disediakan dalam proyek ini tampak lengkap, dalam skenario data historis yang lebih besar atau dari sumber yang berbeda, nilai hilang seringkali menjadi masalah. Jika nilai hilang terdeteksi, strategi penanganannya akan ditentukan setelah eksplorasi data. Pendekatan umum untuk data deret waktu meliputi forward-fill (mengisi nilai hilang dengan nilai terakhir yang diketahui) atau penghapusan baris/kolom jika proporsi nilai hilang sangat kecil dan tidak berdampak signifikan pada dataset.
    - Justifikasi: Nilai hilang dapat menyebabkan error pada model selama pelatihan atau menghasilkan bias yang tidak diinginkan dalam proses pembelajaran, yang pada akhirnya dapat menurunkan kinerja model.
- Pembentukan Fitur (Feature Engineering):
    - Ini adalah langkah kunci untuk meningkatkan informasi yang dapat diekstrak dari data mentah.
        - Lagged Features: Fitur-fitur baru akan dibuat dari harga penutupan (Terakhir), volume (Vol.), dan persentase perubahan (Perubahan%) dari hari-hari perdagangan sebelumnya (misalnya, Terakhir_t-1, Vol_t-1, Perubahan%_t-1). Ini esensial untuk menangkap dependensi temporal dan dinamika pasar dari waktu ke waktu.
        - Moving Averages: Rata-rata bergerak sederhana atau eksponensial (misalnya, rata-rata 5-hari atau 10-hari) dari harga akan dihitung. Indikator ini membantu mengidentifikasi tren jangka pendek dan menghaluskan fluktuasi harga harian.
        - Indikator Teknis Lainnya: Meskipun tidak secara eksplisit ada dalam data mentah, indikator teknis seperti Relative Strength Index (RSI) atau Moving Average Convergence Divergence (MACD) dapat dihitung dari fitur harga yang ada. Indikator-indikator ini memberikan sinyal momentum dan kondisi overbought/oversold yang relevan dalam analisis pasar saham.
    - Justifikasi: Fitur-fitur ini membantu model memahami konteks historis dan dinamika pasar yang lebih dalam. Dengan mengubah data mentah menjadi representasi yang lebih informatif, model dapat belajar pola yang lebih kompleks dan menghasilkan prediksi yang lebih akurat.
- Pembentukan Variabel Target:
    - Variabel target biner, Arah_Gerak, akan dibuat berdasarkan kolom Perubahan%. Jika Perubahan% > 0, Arah_Gerak akan diberi nilai 1 (harga naik); jika Perubahan% <= 0, Arah_Gerak akan diberi nilai 0 (harga turun atau stagnan).
    - Justifikasi: Mengubah masalah prediksi nilai harga (regresi) menjadi klasifikasi arah pergerakan harga adalah pendekatan yang lebih praktis untuk mendukung keputusan investasi harian. Ini juga selaras dengan fokus pada tugas klasifikasi yang relevan dengan penelitian yang tersedia, seperti analisis sentimen.   
- Pembagian Data:
    - Data akan dibagi menjadi set pelatihan (training set), set validasi (validation set), dan set pengujian (test set) secara kronologis. Pembagian ini penting untuk data deret waktu untuk mencegah data leakage, yaitu situasi di mana informasi masa depan bocor ke dalam data pelatihan. Misalnya, 80% data awal akan digunakan untuk pelatihan, 10% berikutnya untuk validasi, dan 10% terakhir untuk pengujian.
    - Justifikasi: Pembagian kronologis memastikan bahwa model hanya dilatih pada informasi yang tersedia di masa lalu, mereplikasi skenario dunia nyata di mana prediksi dibuat berdasarkan data historis yang tersedia.
- Normalisasi/Standardisasi Fitur:
    - Fitur-fitur numerik, seperti harga, volume, dan persentase perubahan (termasuk fitur yang direkayasa), akan dinormalisasi atau distandardisasi. Metode umum yang digunakan adalah StandardScaler (mengubah data menjadi memiliki rata-rata 0 dan standar deviasi 1) atau MinMaxScaler (menskalakan data ke rentang tertentu, misalnya 0 hingga 1).
    - Justifikasi: Banyak algoritma machine learning, terutama yang berbasis jarak atau gradien, sangat sensitif terhadap skala fitur. Normalisasi membantu mencegah fitur dengan rentang nilai yang besar mendominasi proses pembelajaran model, mempercepat konvergensi algoritma, dan secara signifikan meningkatkan kinerja model.

Dengan menerapkan teknik-teknik persiapan data ini secara cermat, data akan siap untuk tahapan pemodelan, memastikan bahwa model dapat belajar dari representasi data yang paling optimal.

## Modeling
Tahapan pemodelan adalah inti dari proyek machine learning, di mana algoritma dipilih, dikonfigurasi, dan dilatih untuk menyelesaikan masalah yang telah didefinisikan. Pemilihan algoritma yang tepat dan optimasi parameternya sangat krusial untuk mencapai kinerja prediksi yang diinginkan.

### Pemilihan Model Machine Learning

Berdasarkan sifat masalah sebagai tugas klasifikasi biner (memprediksi apakah harga saham akan naik atau turun) dan karakteristik data deret waktu, beberapa algoritma machine learning yang relevan akan dipertimbangkan dan diimplementasikan:

- Logistic Regression:
    - Kelebihan: Model ini relatif sederhana, cepat untuk dilatih, dan hasil prediksinya mudah diinterpretasikan dalam bentuk probabilitas. Algoritma ini berfungsi sebagai baseline yang sangat baik untuk membandingkan kinerja model yang lebih kompleks.   
    - Kekurangan: Asumsi utama dari Logistic Regression adalah adanya hubungan linear antara fitur input dan log-odds dari variabel target. Hal ini membuatnya kurang efektif dalam menangkap pola non-linear yang kompleks dan interaksi antar fitur yang sering ditemukan dalam data pasar saham yang dinamis.
- Support Vector Machine (SVM):
    - Kelebihan: SVM sangat efektif dalam ruang berdimensi tinggi dan memiliki kemampuan untuk menangani hubungan non-linear melalui penggunaan kernel trick. Ini memungkinkan SVM untuk menemukan hyperplane optimal yang memisahkan kelas-kelas dengan margin terbesar, sehingga menghasilkan kemampuan generalisasi yang baik, bahkan pada data yang tidak terlihat sebelumnya. SVM sangat cocok untuk masalah klasifikasi biner.   
    - Kekurangan: Kinerja SVM sangat sensitif terhadap penskalaan fitur, sehingga normalisasi data menjadi prasyarat. Selain itu, pemilihan hyperparameter yang tepat (terutama jenis kernel dan parameter regularisasi C) memerlukan tuning yang cermat. SVM juga cenderung kurang efisien secara komputasi untuk dataset yang sangat besar.
- Random Forest:
    - Kelebihan: Random Forest adalah metode ensemble yang kuat, dibangun dari banyak decision tree individu. Algoritma ini mampu menangani hubungan non-linear dan interaksi fitur secara alami. Keunggulan utamanya adalah ketahanannya terhadap overfitting (dibandingkan dengan decision tree tunggal) dan kemampuannya untuk memberikan estimasi pentingnya fitur, yang dapat memberikan wawasan tentang faktor-faktor pendorong prediksi. Random Forest juga cenderung berkinerja baik pada data dengan banyak fitur.   
    - Kekurangan: Salah satu kelemahan utama Random Forest adalah sifatnya yang kurang interpretable atau sering disebut sebagai black-box model, karena melibatkan kombinasi dari banyak pohon keputusan. Meskipun demikian, ini adalah kompromi yang sering diterima mengingat kinerja prediktifnya yang kuat. Selain itu, model ini bisa menjadi intensif secara komputasi dan memori jika jumlah pohon yang digunakan sangat banyak atau dataset sangat besar.

### Tahapan Pemodelan dan Konfigurasi Parameter
Setelah pemilihan algoritma, tahapan pemodelan akan melibatkan langkah-langkah berikut:

- Inisialisasi Model: Setiap algoritma yang dipilih (Logistic Regression, SVM, Random Forest) akan diinisialisasi, awalnya dengan parameter default atau parameter awal yang umum digunakan.
- Pelatihan Model Awal: Model-model ini akan dilatih menggunakan set pelatihan yang telah disiapkan pada tahap persiapan data.
- Evaluasi Baseline: Kinerja awal setiap model akan dievaluasi pada set validasi untuk mendapatkan gambaran baseline performance. Ini membantu mengidentifikasi model mana yang memiliki potensi lebih besar sebelum optimasi lebih lanjut.

### Proses Peningkatan Model atau Pemilihan Model Terbaik
Untuk memaksimalkan potensi kinerja model dan memastikan keandalannya, proses optimasi dan pemilihan model terbaik akan dilakukan:

- Hyperparameter Tuning:
    - Setiap algoritma memiliki hyperparameter yang memengaruhi proses pembelajarannya dan, pada akhirnya, kinerjanya. Teknik seperti Grid Search atau Randomized Search yang dikombinasikan dengan Cross-Validation akan diterapkan untuk menemukan kombinasi hyperparameter optimal untuk setiap model.
    - Contoh hyperparameter yang akan disetel:
        - Logistic Regression: Kekuatan regularisasi (C) untuk mengontrol overfitting.
        - SVM: Jenis kernel (linear, RBF), parameter regularisasi C, dan gamma (untuk kernel RBF) yang memengaruhi bentuk hyperplane dan sensitivitas model terhadap titik data.
        - Random Forest: Jumlah pohon dalam ensemble (n_estimators), kedalaman maksimum setiap pohon (max_depth), dan jumlah sampel minimum yang diperlukan untuk membagi sebuah node (min_samples_leaf).
    - Justifikasi: Hyperparameter tuning adalah proses iteratif yang sangat penting untuk memaksimalkan potensi kinerja model. Tanpa tuning yang tepat, model dapat mengalami underfitting (tidak cukup kompleks untuk menangkap pola dalam data) atau overfitting (terlalu kompleks dan hanya menghafal data pelatihan, sehingga buruk dalam generalisasi ke data baru). Proses ini bertujuan untuk menemukan konfigurasi model yang paling sesuai dengan karakteristik data dan masalah yang dihadapi.
- Pemilihan Model Terbaik:
    - Setelah hyperparameter tuning selesai untuk setiap algoritma, kinerja model yang telah dioptimalkan akan dibandingkan secara ketat menggunakan metrik evaluasi yang telah ditentukan (Akurasi, Presisi, Recall, F1-Score) pada set validasi.
    - Justifikasi Pemilihan Model Terbaik: Model dengan F1-Score tertinggi akan dipilih sebagai model terbaik. F1-Score dipilih karena memberikan keseimbangan yang baik antara Presisi dan Recall. Dalam konteks prediksi pasar saham, di mana baik false positives (memprediksi kenaikan padahal turun, yang dapat menyebabkan kerugian) maupun false negatives (memprediksi penurunan padahal naik, yang berarti kehilangan peluang) sama-sama memiliki konsekuensi finansial, F1-Score adalah metrik yang sangat relevan. Metrik ini memastikan bahwa model tidak hanya akurat dalam prediksi positifnya, tetapi juga cukup komprehensif dalam menangkap sebagian besar pergerakan harga ke atas yang sebenarnya. Model yang terpilih ini kemudian akan dievaluasi secara final pada set pengujian yang belum pernah dilihat sebelumnya untuk mendapatkan estimasi kinerja yang paling tidak bias.

## Evaluation
Bagian evaluasi ini menyajikan metrik yang digunakan untuk mengukur kinerja model machine learning dan menginterpretasikan hasilnya dalam konteks bisnis. Pemilihan metrik yang tepat dan pemahaman mendalam tentang implikasinya sangat penting untuk menilai keandalan dan utilitas model.

### Metrik Evaluasi yang Digunakan
Untuk proyek klasifikasi prediksi arah pergerakan harga saham ini, metrik evaluasi yang akan digunakan adalah:
- Akurasi (Accuracy)
- Presisi (Precision)
- Recall (Sensitivity/True Positive Rate)
- F1-Score

Metrik-metrik ini dipilih karena secara kolektif memberikan gambaran yang komprehensif tentang kinerja model dalam tugas klasifikasi biner, khususnya dalam konteks di mana keseimbangan antara berbagai jenis kesalahan prediksi memiliki implikasi finansial yang signifikan.

### Penjelasan Mengenai Metrik yang Digunakan
Memahami setiap metrik adalah kunci untuk menginterpretasikan hasil evaluasi model secara benar:

- Akurasi (Accuracy):
    - Penjelasan: Akurasi mengukur proporsi total prediksi yang benar (baik prediksi positif yang benar maupun prediksi negatif yang benar) dari keseluruhan jumlah observasi. Ini adalah metrik yang paling intuitif dan sering digunakan sebagai indikator kinerja umum.
    - Formula: (True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives)
    - Cara Kerja: Metrik ini memberikan gambaran seberapa sering model membuat prediksi yang tepat. Namun, untuk dataset yang tidak seimbang (misalnya, jumlah hari harga stagnan/turun jauh lebih banyak daripada hari harga naik), akurasi saja bisa menyesatkan. Model yang hanya memprediksi kelas mayoritas bisa menunjukkan akurasi tinggi meskipun tidak berguna.

- Presisi (Precision):
    - Penjelasan: Presisi mengukur proporsi prediksi positif yang benar dari semua instance yang diprediksi sebagai positif oleh model. Dalam konteks proyek ini, presisi menunjukkan seberapa sering model benar ketika memprediksi bahwa harga saham akan naik.
    - Formula: True Positives / (True Positives + False Positives)
    - Cara Kerja: Presisi sangat penting ketika biaya dari false positive (yaitu, model memprediksi harga akan naik, tetapi ternyata turun, yang dapat menyebabkan kerugian investasi jika investor membeli berdasarkan sinyal palsu) sangat tinggi. Investor yang konservatif mungkin lebih menghargai presisi tinggi.

- Recall (Sensitivity/True Positive Rate):
    - Penjelasan: Recall mengukur proporsi instance positif yang benar-benar diidentifikasi oleh model dari semua instance positif yang sebenarnya ada dalam dataset. Dalam konteks ini, recall menunjukkan seberapa baik model menangkap semua kejadian di mana harga saham benar-benar naik.
    - Formula: True Positives / (True Positives + False Negatives)
    - Cara Kerja: Recall penting ketika biaya dari false negative (yaitu, model memprediksi harga akan turun atau stagnan, tetapi ternyata naik, yang berarti kehilangan peluang keuntungan) sangat tinggi. Investor yang ingin memaksimalkan peluang keuntungan mungkin lebih menghargai recall tinggi.

- F1-Score:
    - Penjelasan: F1-Score adalah rata-rata harmonik dari Presisi dan Recall. Ini adalah metrik yang lebih seimbang, terutama ketika ada ketidakseimbangan kelas atau ketika Presisi dan Recall sama-sama penting dalam konteks masalah.
    - Formula: 2 * (Presisi * Recall) / (Presisi + Recall)
    - Cara Kerja: F1-Score memberikan skor tunggal yang menyeimbangkan kemampuan model untuk tidak membuat prediksi positif yang salah (Presisi) dan kemampuannya untuk menemukan semua instance positif (Recall). Dalam prediksi saham, di mana baik kerugian akibat prediksi salah maupun hilangnya peluang sama-sama merugikan, F1-Score adalah metrik yang sangat relevan dan seringkali menjadi pilihan utama untuk model klasifikasi.

### Hasil Evaluasi:

**Sebelum Tuning**:

**Logistic Regression**
- Akurasi: 56%
- Recall kelas 1 (naik): 0.00 (model tidak mendeteksi kenaikan)
- F1-score kelas 1: 0.01

**SVM**
- Akurasi: 56%
- Recall kelas 1: 0.02
- F1-score kelas 1: 0.03

**Random Forest**
- Akurasi: 55%
- Recall kelas 1: 0.52
- Precision kelas 1: 0.50
- F1-score kelas 1: 0.51

**Setelah Tuning**:

**Logistic Regression (Tuning)**
- Akurasi: 56%
- Recall kelas 1 (naik): 0.00 (model tidak mendeteksi kenaikan)
- F1-score kelas 1: 0.00

**SVM (Tuning)**
- Akurasi: 56%
- Recall kelas 1: 0.00
- F1-score kelas 1: 0.00

**Random Forest (Tuning)**
- Akurasi: 56%
- Recall kelas 1: 0.54
- Precision kelas 1: 0.13
- F1-score kelas 1: 0.21

Bisa terlihat berdasarkan Confusion Matrix bahwa Logistic Regression dan SVM mengalami kegagalan dalam mengenali kelas naik meskipun akurasinya terlihat seimbang. Sementara Random Forest mampu menunjukkan kemampuan awal mengenali kedua kelas, meskipun hasilnya tetap kurang baik dan tetap cenderung bias terhadap kelas mayoritas.

Sehingga yang bisa disimpulkan dari proyek ini adalah semua model yang dibangun seperti model Logistic Regression , model SVM, dan Random Forest masih sangat buruk performanya. Model - model menunjukkan keterbatasan dalam mendeteksi kelas 1 (harga naik), bahkan setelah dilakukan tuning sekalipun. Meskipun memang model Random Forest menjadi model paling seimbang diantara yang lain, namun recall dan f1-score masih rendah dan belum memenuhi ekspektasi yang diharapkan. Hal yang menjadi menjadi pengaruh yang sangat besar mungkin disbebabkan karena dataset sangat dipengaruhi oleh class imbalance dan fitur terbatas.

Hal - hal yang mungkin bisa diterapkan berikutnya agar prediksi yang dihasilkan lebih baik adalah:
1. Menggunakan teknik class balancing seperti SMOTE atau class_weight=balanced bisa menjadi salah satu cara yang bisa dilakukan.
2. Fitur teknikal tambahan seperti Bollinger Bands, Stochastic Oscillator bisa ditambahkan guna untuk menghasilkan hasil prediksi yang lebih baik.
3. Bisa melakukan evaluasi model menggunakan strategi trading simulasi untuk menilai dampak praktis dari prediksi.
4. Mempertimbangkan fitur berbasis berita atau sentimen pasar untuk memperkaya konteks prediktif agar prediksi yang dihasilkan semakin baik.


**---Ini adalah bagian akhir laporan---**
