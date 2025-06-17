# Laporan Proyek Machine Learning - Ryanza Aufa Yansa

## Domain Proyek (Bisnis)
Film merupakan salah satu bentuk hiburan visual yang memiliki daya tarik tinggi di kalangan masyarakat. Seiring dengan pesatnya pertumbuhan industri film di era digital, jumlah judul yang tersedia di pasaran meningkat secara signifikan. Kondisi ini menciptakan tantangan bagi konsumen dalam memilih konten yang sesuai dengan preferensi mereka. Untuk menjawab kebutuhan tersebut, dikembangkanlah berbagai solusi teknologi, salah satunya melalui penerapan sistem rekomendasi guna meningkatkan pengalaman pengguna dan efisiensi dalam penemuan konten. Sistem rekomendasi dirancang untuk menyajikan item yang paling relevan dan berpotensi menarik bagi pengguna tertentu. Dengan memanfaatkan data interaksi historis dari pengguna, sistem ini mampu memprediksi konten atau produk yang sesuai dengan preferensi individu. Dalam konteks ini, sistem rekomendasi difokuskan pada penyajian judul film yang memiliki kemiripan atau keterkaitan dengan konten yang sebelumnya diminati oleh pengguna [1]. Dalam proyek ini, dibangun dua pendekatan sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering [2][3], untuk merekomendasikan film berdasarkan informasi yang tersedia dari dataset [Movie Recommender System](https://www.kaggle.com/datasets/ranitsarkar01/movies-recommender-system-dataset).

Pentingnya proyek ini terletak pada kemampuannya menyajikan rekomendasi yang dipersonalisasi kepada pengguna, meningkatkan keterlibatan, dan memberikan alternatif tontonan yang relevan. Dengan semakin melimpahnya data film dan ulasan pengguna, sistem rekomendasi menjadi krusial untuk mengelola informasi tersebut secara efektif.

## Business Understanding

### Problem Statements
Rumusan masalah yang bisa didapatkan:
1. Bagaimana menyarankan film yang relevan kepada pengguna berdasarkan title dan genre film yang telah disukai sebelumnya?
2. Bagaimana membangun model rekomendasi yang mampu memahami pola interaksi antara pengguna dan film menggunakan data rating?

### Goals:
Tujuan yang bisa didapatkan dari rumusan masalah:
1. Mengembangkan sistem rekomendasi berbasis konten yang mampu menyarankan film serupa berdasarkan title dan genre film.
2. Membangun dan melatih model collaborative filtering berbasis neural network yang dapat memberikan rekomendasi film personal kepada pengguna berdasarkan data interaksi rating.

### Solution Approach
1. Menerapkan teknik TF-IDF vectorization dan cosine similarity untuk menghasilkan rekomendasi berdasarkan kemiripan konten antar film.
2. Membangun arsitektur RecommenderNet (embedding layer) menggunakan TensorFlow untuk belajar dari embedding pengguna dan film, serta memprediksi rating secara akurat menggunakan metrik evaluasi MAE dan Loss.

## Data Understanding
Dataset yang digunakan untuk membangun sistem rekomendasi diambil dari [**Kaggle: Movie Recommender System**](https://www.kaggle.com/datasets/ranitsarkar01/movies-recommender-system-dataset). Dataset tersebut dipublikasikan oleh Ranit Sarkar. Dataset ini berisi informasi mengenai film dan interaksi pengguna dalam bentuk rating, serta metadata terkait film seperti genre. Dataset ini sangat cocok untuk membangun sistem rekomendasi baik berbasis konten (content-based) maupun kolaboratif (collaborative filtering). Dataset ini memiliki tingkat *usability* mencapai 10.00/10.00. Dataset terdiri dari 3 file DAT tetapi yang akan digunakan hanya 2, yaitu:
- movies.dat
- ratings.dat

### *Exploratory Data Analysis* (EDA)

#### Memahami Struktur Data

**Movies.dat**
  
![dataset_movie](https://github.com/user-attachments/assets/55dfa414-7d54-4261-9356-286934ab2654)

Dataset memuat total **3883 baris data** dan **3 kolom** diawal. Berikut uraian 3 kolom:
- **movieId**: Berisikan **ID unik film** yang merepresentasikan setiap film dalam dataset.
- **Title**: Berisikan **judul lengkap film** beserta tahun rilis dalam tanda kurung.
- **Genres**: Berisikan **kategori/genre film** dalam bentuk string yang dipisahkan dengan tanda **|**.

**Ratings.dat**

![dataset_ratings](https://github.com/user-attachments/assets/ab798250-9232-4ce1-9733-c693e4acf349)

Dataset memuat total **1000209 baris data** dan **4 kolom** diawal. Berikut uraian 4 kolom:
- **userId**: Berisikan **ID unik pengguna** yang memberikan penilaian terhadap film.
- **movieId**: Berisikan **ID unik film** yang telah diberi penilaian.
- **Rating**: Berisikan **skor penilaian** yang diberikan oleh pengguna, dalam skala 1 sampai 5.
- **timestamp**: Berisikan **waktu** saat penilaian diberikan, dalam format timestamp Unix (epoch time), bisa dikonversi ke tanggal-waktu.

#### Mengidentifikasi Missing dan Duplicate Values

![identifikasi_miss dupplicate](https://github.com/user-attachments/assets/ad67cbd8-661a-4de7-b92c-0d880ed5e37c)

Dataset tidak memiliki **missing values** dan **duplicate values**.

#### Analisis Deskriptif dan Univariate Analysis

**Movies.dat**

| Statistik | movieId     |
|-----------|-------------|
| count     | 3,883       |
| mean      | 1,986.049   |
| std       | 1,146.778   |
| min       | 1           |
| 25%       | 982.500     |
| 50%       | 2,010.000   |
| 75%       | 2,980.500   |
| max       | 3,952.000   |

Berikut informasi yang didapat dari informasi deskriptif tersebut:
- Dataset berisikan **3883 film**
- Informasi deskriptif yang ada pada dataset **movies** hanyalah informasi **ID**. ID film terkecil adalah 1 dan terbesar adalah 3952.

**Ratings.dat**

| Statistik | userId       | movieId      | Rating       | timestamp     |
|-----------|--------------|--------------|--------------|---------------|
| count     | 1,000,209    | 1,000,209    | 1,000,209    | 1,000,209     |
| mean      | 3,024.512    | 1,865.540    | 3.581        | 972,243,700   |
| std       | 1,728.413    | 1,096.041    | 1.117        | 12,152,560    |
| min       | 1            | 1            | 1.0          | 956,703,900   |
| 25%       | 1,506        | 1,030        | 3.0          | 965,302,600   |
| 50%       | 3,070        | 1,835        | 4.0          | 973,018,000   |
| 75%       | 4,476        | 2,770        | 4.0          | 975,220,900   |
| max       | 6,040        | 3,952        | 5.0          | 1,046,455,000 |

Berikut informasi yang didapat dari informasi deskriptif tersebut:
- Dataset ratings berisi **1.000.209** data dari para pengguna.
- Pada kolom **userId**, terdapat **6040** pengguna unik.
- Pada kolom **movieID**, film yang dinilai memiliki ID dari 1 hingga 3952.
- Pada kolom **Rating**, rating berkisar dari 1 **(terendah)** sampai 5 **(tertinggi)**. **Rata-rata rating** yang diberikan adalah **3.58**, menunjukkan kecenderungan pengguna memberikan **rating positif**. **Median rating (nilai tengah)** adalah **4.0**, menunjukkan bahwa sebagian besar pengguna memberikan **rating baik**.

Berikut beberapa visualisasi persebaran data

![distribusi_rating](https://github.com/user-attachments/assets/9062eb4a-88d7-446a-9318-a260d5239c67)
![distribusi_jumlah_rating](https://github.com/user-attachments/assets/c8d80f3b-e55a-4111-a570-c22675064b1a)
![film_rating_terbanyak](https://github.com/user-attachments/assets/9474d826-240c-4fe1-bd76-3cf30930f53a)
![distribusi_jumlah_rating_per_pengguna](https://github.com/user-attachments/assets/377819ff-dd05-4ce3-8749-43929903c71e)
![distribusi_jumlah_film_per_genre](https://github.com/user-attachments/assets/3ede6864-deb8-4134-a035-0e2e76e6eb99)

Berikut informasi yang didapat dari grafik persebaran data:
1. **Distribusi Rating Pengguna:**

   - Distribusi rating pengguna **cenderung terpusat di rating 4 dan 5**, menunjukkan bahwa sebagian besar pengguna memberikan penilaian positif terhadap film yang mereka tonton.
   - Rating rendah (1–2) memiliki jumlah yang jauh lebih sedikit, yang mengindikasikan tingkat kepuasan pengguna relatif tinggi.

2. **Distribusi Jumlah Rating per Film:**

   - Sebagian besar film menerima **jumlah rating yang rendah**, sementara hanya sedikit film yang mendapatkan lebih dari 3000 rating.
   - Ini menunjukkan adanya **ketimpangan popularitas** antar film: beberapa film sangat populer sementara mayoritas hanya mendapatkan perhatian terbatas.

3. **10 Film dengan Jumlah Rating Terbanyak:**

   - Film seperti *American Beauty (1999)*, *Star Wars* series, *The Matrix (1999)*, dan *Saving Private Ryan (1998)* menjadi yang paling banyak mendapat rating.
   - Film-film ini termasuk dalam genre populer dan memiliki pengaruh budaya besar, yang mungkin menjelaskan jumlah rating yang tinggi.

4. **Distribusi Jumlah Rating per Pengguna:**

   - Sebagian besar pengguna memberikan rating dalam jumlah **rendah hingga menengah**, dengan sedikit pengguna yang sangat aktif (memberikan ratusan hingga ribuan rating).
   - Hal ini menunjukkan bahwa dataset memiliki **long-tail behavior**, di mana sebagian besar kontribusi berasal dari sedikit pengguna aktif.

5. **Jumlah Film per Genre:**

   - Genre **Drama** mendominasi dengan jumlah film terbanyak, diikuti oleh **Comedy**, **Action**, dan **Thriller**.
   - Genre seperti **Film-Noir**, **Western**, dan **Fantasy** memiliki jumlah film yang jauh lebih sedikit, mencerminkan mungkin preferensi industri atau minat pasar yang terbatas.

#### Multivariate Analysis
Berikut visualisasi untuk rata-rata rating per tahun rilis dan genre:
![mean_rating_per_year](https://github.com/user-attachments/assets/2147dbb2-9573-478b-a707-96dc7c0f7b8a)
![mean_rating_per_genre](https://github.com/user-attachments/assets/5d88344e-5699-4789-8c22-ca474b027456)

1. **Rata-rata Rating per Year (Tahun Rilis Film):**

   - Rata-rata rating film dari tahun ke tahun **relatif stabil**, dengan sedikit fluktuasi.
   - Film-film dari era **1970-an hingga awal 2000-an** menunjukkan rata-rata rating yang sedikit lebih tinggi dibandingkan film-film dari era 1950-an.
   - Hal ini bisa menunjukkan adanya **sentimen nostalgia atau kualitas klasik** pada film-film lama, serta **penurunan minat atau kualitas yang dirasakan** pada beberapa film baru.
2. **Rata-rata Rating per Genre:**

   - Genre dengan rata-rata rating tertinggi adalah **Film-Noir**, **Documentary**, dan **War**, menunjukkan bahwa meskipun film dalam genre ini lebih sedikit, kualitas persepsi dari penonton cukup tinggi.
   - Genre dengan rata-rata rating lebih rendah adalah **Horror** dan **Children’s**, yang mungkin disebabkan oleh ekspektasi atau variasi kualitas dalam masing-masing genre.

## Data Preparation
Tahap ini mencakup pembersihan dan transformasi data agar siap digunakan dalam pemodelan sistem rekomendasi

### Merged kedua dataframe dan mengubah beberapa fitur
Pada tahap ini kedua dataset digabung menjadi 1 berdasarkan fitur movieId. Hal ini dilakukan agar dapat mempermudah membagi data yang diperlukan untuk *content based filtering* dan *collaborative based filtering*

![merged_dataset](https://github.com/user-attachments/assets/d49e9f30-0af2-4950-be20-005ee4a092f7)


### Memisahkan data untuk content based dan collaborative based
Dataset yang telah dimerged dipisahkan untuk masing-masing keperluan sistem rekomendasi. Nantinya kedua pendekatan akan menggunakan dataset yang berbeda agar hasil diberikan lebih baik. Dataset juga akan diproses sesuai kebutuhan kedua pendekatan tersebut.

- *Content Based Filtering*

  Berikut dataset awal untuk *content based*. Data untuk pendeketan ini berfokus kepada title, year dan genre

  ![drop_user_rating](https://github.com/user-attachments/assets/0e508516-f30e-4a58-a559-619b29957ef4)
  
  ![membuat_features](https://github.com/user-attachments/assets/b761b3a1-4385-4462-9453-d864136d0d86)
  
  ![dataset_content_based](https://github.com/user-attachments/assets/7fa3a962-32e4-40a5-998b-10659ca8aeaa)

  Selanjutnya dataset tersebut akan diproses agar bisa menghasilkan matriks yang nantinya akan digunakan untuk proses modeling sistem rekomendasi *content based* dengan *cosine  similiarity*.

  ![init_tfidf](https://github.com/user-attachments/assets/457e19ee-ccbc-4b7b-989e-944766647098)

  ![dataframe_matriks_content_based](https://github.com/user-attachments/assets/dd748b66-f002-43e0-84dc-e4964988125e)

- *Collaborative Based Filtering*

  Berikut dataset awal untuk *collaborative based*. Dataset untuk pendekatan ini akan berfokus kepada informasi users seperti ratings dan userId. Data ini sudah melalui proses encoding untuk movie dan user id serta normalisasi untuk rating.

  **Drop Kolom Year dan Genre**

  ![drop_year_genre](https://github.com/user-attachments/assets/3bb3e89b-1179-4eb7-868b-2527a7a2ae51)

  **Encoding dan Normalisasi**

  ![encoding_normalisasi](https://github.com/user-attachments/assets/275383d0-ea6b-4438-87b5-f0ddee9c4575)

  **Dataset yang telah diproses**
  
  ![processed_dataset_collab_based](https://github.com/user-attachments/assets/3a64cc4e-ed9f-4d14-adee-e62ae85c9fd7)

## Modeling

### 1. *Content Based Filtering*
   
  Content-Based Filtering pada sistem rekomendasi ini memanfaatkan algoritma cosine similarity untuk mengukur tingkat kemiripan antar item berdasarkan informasi konten, seperti genre dan deskripsi film. Cosine similarity bekerja dengan cara menghitung sudut kosinus antara dua vektor dalam ruang berdimensi tinggi untuk menentukan seberapa mirip arah kedua vektor tersebut. Semakin kecil sudut antar vektor, semakin besar nilai cosine similarity-nya, yang menandakan bahwa kedua item tersebut memiliki karakteristik yang serupa [4].

$$Cos (\theta) = \frac{\sum_1^n a_ib_i}{\sqrt{\sum_1^n a_i^2}\sqrt{\sum_1^n b_i^2}}$$

  Secara matematis, cosine similarity dirumuskan sebagai perbandingan antara hasil perkalian dot product dua vektor dengan hasil kali dari norma (magnitudo) masing-masing vektor. Dalam implementasinya di Python, fungsi cosine_similarity dari pustaka sklearn.metrics.pairwise digunakan untuk menghitung nilai kemiripan antar vektor dalam sebuah matriks, seperti matriks TF-IDF.

  Kelebihan dari cosine similarity meliputi hasil output yang telah ternormalisasi (dalam rentang -1 hingga 1), interpretasi yang intuitif, serta efisiensi dalam menangani data sparse berdimensi tinggi. Namun, metode ini juga memiliki beberapa keterbatasan, seperti asumsi bahwa semua fitur memiliki bobot yang sama, kepekaan terhadap perubahan arah vektor, dan ketidaksesuaian jika diterapkan pada data dengan nilai negatif.

  Setelah sistem rekomendasi dibangun menggunakan pendekatan ini, langkah berikutnya adalah melakukan pengujian dengan menampilkan top 10 rekomendasi film berdasarkan konten yang relevan. Rekomendasi disesuaikan dengan input film dari pengguna dan dipilah berdasarkan atribut seperti genre, title dan year yang paling mendekati preferensi pengguna. Berikut hasil yang didapat.

![hasil_content_based](https://github.com/user-attachments/assets/9229284d-6076-47f9-baf2-f07a75cda779)

### 2. *Collaborative Based Filtering*

Collaborative Filtering dalam proyek ini diimplementasikan menggunakan pendekatan deep learning, khususnya dengan memanfaatkan embedding layer sebagai inti dari arsitektur model. Embedding layer merupakan jenis lapisan dalam jaringan saraf tiruan yang berfungsi untuk mengonversi data kategorikal (seperti userId dan movieId) menjadi representasi vektor berdimensi kontinu. Representasi ini memungkinkan model untuk menangkap hubungan laten atau latent factors antar pengguna dan film dalam ruang vektor [4].

Dalam implementasinya menggunakan Python, embedding layer dibentuk melalui modul tensorflow.keras.layers.Embedding, yang secara efisien menghasilkan representasi numerik dari masing-masing entitas kategorikal. Pendekatan ini memiliki sejumlah kelebihan, di antaranya: mengurangi kompleksitas model dengan memetakan ID ke ruang vektor berdimensi lebih rendah, fleksibel untuk digunakan pada berbagai arsitektur deep learning, serta mampu menangkap hubungan semantik dan preferensi implisit antara pengguna dan item.

Namun, embedding layer juga memiliki beberapa keterbatasan. Di antaranya adalah ketergantungan terhadap jumlah data yang besar untuk mencapai performa optimal, sensitivitas terhadap pemilihan hyperparameter (seperti dimensi embedding, learning rate, dan jumlah epoch), serta cold start problem ketika menghadapi user atau item baru yang belum memiliki histori interaksi.

Setelah proses pelatihan model selesai, diperoleh hasil evaluasi menggunakan metrik loss dan mean absolute error. Berdasarkan model collaborative filtering yang telah dibangun, sistem kemudian diuji untuk menghasilkan top 10 rekomendasi film yang paling relevan untuk masing-masing pengguna. Rekomendasi ini disusun berdasarkan pola interaksi dan preferensi pengguna terhadap film-film sebelumnya, tanpa melihat isi konten film secara langsung.

![hasil_collab_based](https://github.com/user-attachments/assets/74c80302-fca1-4649-82f9-eaf4ae96ac33)


## Evaluation

### 1. *Content Based Filtering*

Dalam pendekatan content-based filtering, evaluasi kinerja model dilakukan dengan menggunakan metrik Precision. Precision merupakan salah satu metrik evaluasi yang umum digunakan dalam sistem rekomendasi, khususnya untuk mengukur sejauh mana rekomendasi yang diberikan model benar-benar relevan bagi pengguna.

Secara definisi, Precision menghitung proporsi item yang relevan (benar-benar sesuai dengan preferensi pengguna) di antara seluruh item yang direkomendasikan. Dengan kata lain, metrik ini menilai seberapa akurat model dalam menyarankan film yang sesuai dengan kebutuhan pengguna, tanpa mempertimbangkan item relevan lain yang mungkin tidak direkomendasikan.

$$ Precision = \frac{TP}{TP + FP} $$

Dimana:

- TP (*True Positive*), jumlah kejadian positif yang diprediksi dengan benar.
- FP (*False Positive*), jumlah kejadian positif yang diprediksi dengan salah.

![precision_content_based](https://github.com/user-attachments/assets/4807dd9e-d3d3-4abd-853c-f82dd1b27a84)

Berdasarkan hasil yang terdapat pada tahap Model and Result dapat dilihat bahwasanya:
- Hasil rekomendasi berdasarkan **genre** berhasil memeberikan rekomendasi yang sesuai **(6 dari 10)**
- Hasil rekomendasi berdasarkan **tahun** berhasil memberikan rekomendasi yang sesuai **(10 dari 10)**

### 2. *Collaborative Based Filtering*

![evaluation_graph](https://github.com/user-attachments/assets/3ddedc2e-7f94-4c25-bb88-6c0b0a2b9f7b)

Berdasarkan hasil pelatihan model collaborative filtering berbasis deep learning selama 10 epoch, performa model menunjukkan hasil yang cukup stabil namun mengalami sedikit overfitting seiring bertambahnya epoch. Hal ini dapat terlihat dari grafik Training vs Validation Loss (MSE) dan Training vs Validation MAE, serta nilai metrik pada setiap epoch.

Pada epoch awal, tepatnya pada epoch ke-2 dan ke-3, model menunjukkan performa terbaiknya dengan nilai val_loss sebesar 0.0682 dan val_mean_absolute_error (MAE) sebesar 0.1991, yang lebih rendah dibandingkan MAE data pelatihan pada beberapa epoch selanjutnya. Setelah mencapai titik optimal tersebut, training loss justru mengalami peningkatan dari 0.0688 menjadi 0.0745, disertai kenaikan training MAE dari 0.1988 menjadi 0.2040. Sementara itu, validation loss dan validation MAE cenderung stagnan pada angka sekitar 0.0697 dan 0.1978.

Kondisi ini mengindikasikan bahwa model mulai kehilangan kemampuan generalisasi dan cenderung mempelajari data pelatihan secara berlebihan (terjadi overfitting ringan). Walau demikian, performa akhir model masih tergolong baik, ditandai dengan val_MAE yang rendah dan stabil di kisaran 0.198, menunjukkan bahwa model cukup andal dalam memprediksi preferensi pengguna dengan tingkat kesalahan yang kecil. Untuk meningkatkan performa lebih lanjut, dapat dipertimbangkan penerapan teknik regularisasi tambahan atau early stopping agar overfitting dapat dicegah lebih awal dan efisiensi pelatihan meningkat.

## Referensi
[1] Prasetyo, E. A. (2022). Implementasi Collaborative Filtering dalam Sistem Rekomendasi Film Menggunakan Algoritma K-Nearest Neighbor. Universitas Tarumanagara. Diakses pada 16 Juni 2025 dari: https://lintar.untar.ac.id/repository/penelitian/buktipenelitian_10390001_7A281222103549.pdf

[2] Siahaan, A., & Syahputra, D. (2023). Penerapan Metode Content-Based Filtering pada Sistem Rekomendasi Film. JATI: Jurnal Mahasiswa Teknik Informatika, 10(2), 127–133. Diakses pada 16 Juni 2025 dari: https://www.ejournal.itn.ac.id/index.php/jati/article/view/13251/7349

[3] UpGrad. (2023). Create Your Own Movie Recommendation System Using Python. Diakses pada 16 Juni 2025 dari: https://www.upgrad.com/blog/create-your-own-movie-recommendation-system-using-python/

[4] Simorangkir, K. (2024). Skincare Recommendation System using Content-Based Filtering and Collaborative Filtering [Repository GitHub]. Diakses pada 16 Juni 2025 dari: https://github.com/kevinsimorangkir2001/skincare_recommendation/tree/main

[5] Dicoding. Diakses pada 22 Mei 2025 dari: https://www.dicoding.com/academies/319-machine-learning-terapan

  
