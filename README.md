# Gallstone Status Prediction - Data Science Pipeline

Repositori ini berisi pipeline lengkap untuk prediksi status batu empedu menggunakan pendekatan multi-modal stacking ensemble. Pipeline mencakup eksplorasi data, preprocessing, feature engineering, dan pemodelan machine learning dengan classifier gabungan.

---

## Daftar Isi

1. [Ringkasan Proyek](#ringkasan-proyek)
2. [Struktur Pipeline](#struktur-pipeline)
3. [Dataset dan Fitur](#dataset-dan-fitur)
4. [Deskripsi Notebook](#deskripsi-notebook)
5. [Alur Eksekusi](#alur-eksekusi)
6. [Persyaratan Sistem](#persyaratan-sistem)
7. [Instalasi dan Penggunaan](#instalasi-dan-penggunaan)

---

## Ringkasan Proyek

Proyek ini mengembangkan sistem prediksi otomatis untuk mengidentifikasi status batu empedu pada pasien berdasarkan profil klinis, biomedis, dan laboratorium. Dengan memanfaatkan tiga modalitas data berbeda (fisik, kimia, dan demografis), sistem menggunakan arsitektur stacking ensemble dua tingkat untuk meningkatkan akurasi prediksi.

Pendekatan ini menggabungkan kekuatan beberapa algoritma machine learning:
- XGBoost untuk analisis fitur fisik (bioimpedansi)
- LightGBM untuk analisis fitur kimia (profil laboratorium)
- Logistic Regression sebagai meta-learner untuk integrasi prediksi

---

## Struktur Pipeline

Pipeline terdiri dari tiga tahap utama yang saling terhubung:

```
[Raw Data] → [01_EDA] → [02_Preprocessing] → [03_Feature_Engineering] → [04_Modeling] → [05_Evaluasi] → [Model Output]
```

Setiap notebook memiliki tanggung jawab spesifik dalam transformasi dan analisis data menuju model prediktif akhir.

---

## Dataset dan Fitur

### Sumber Data
Dataset UCI Medical (dataset-uci.csv) yang mencakup 319 pasien dengan 38 fitur klinis.

### Kategorisasi Fitur

Dataset dibagi menjadi tiga modalitas utama sesuai dengan sifat dan sumber pengukuran:

#### 1. Fitur Demografis (7 fitur)
- Age
- Gender
- Comorbidity
- Coronary Artery Disease (CAD)
- Hypothyroidism
- Hyperlipidemia
- Diabetes Mellitus (DM)

#### 2. Fitur Fisik - Bioimpedansi (18 fitur)
Pengukuran langsung komposisi tubuh menggunakan teknik bioimpedansi elektrik:
- Height, Weight
- Body Mass Index (BMI)
- Total Body Water (TBW)
- Extracellular Water (ECW)
- Intracellular Water (ICW)
- Extracellular Fluid/Total Body Water (ECF/TBW)
- Total Body Fat Ratio (TBFR)
- Lean Mass (LM)
- Body Protein Content (Protein)
- Visceral Fat Rating (VFR)
- Bone Mass (BM)
- Muscle Mass (MM)
- Obesity (%)
- Total Fat Content (TFC)
- Visceral Fat Area (VFA)
- Visceral Muscle Area (VMA)
- Hepatic Fat Accumulation (HFA)

#### 3. Fitur Kimia - Laboratorium (13 fitur)
Hasil pengukuran dari tes darah dan pemeriksaan laboratorium:
- Glucose
- Total Cholesterol (TC)
- Low Density Lipoprotein (LDL)
- High Density Lipoprotein (HDL)
- Triglyceride
- Aspartat Aminotransferaz (AST)
- Alanin Aminotransferaz (ALT)
- Alkaline Phosphatase (ALP)
- Creatinine
- Glomerular Filtration Rate (GFR)
- C-Reactive Protein (CRP)
- Hemoglobin (HGB)
- Vitamin D

### Target Variabel
**Gallstone Status** - Variabel biner yang mengindikasikan ada/tidaknya batu empedu pada pasien.

---

## Deskripsi Notebook

### 1. 01_EDA.ipynb — Exploratory Data Analysis

Notebook ini berfokus pada eksplorasi menyeluruh terhadap dataset klinis untuk memahami karakteristik, pola distribusi, kualitas data, dan hubungan multivariat awal antar-fitur.

**Tahapan Utama:**
* **Loading & Inspeksi Data:** Membaca dataset utama (`dataset-uci.csv`), memeriksa dimensi, tipe data (numerik vs. kategorikal), dan struktur dasar dataset.
* **Analisis Kualitas Data:** Mendeteksi nilai yang hilang (*missing values*), memeriksa rekaman duplikat, dan mengevaluasi kelengkapan data klinis.
* **Statistik Deskriptif:** Menampilkan ringkasan statistik (*mean*, *median*, standar deviasi, dan kuartil) untuk fitur-fitur medis utama guna mendeteksi pencilan (*outliers*).
* **Visualisasi Distribusi:** Pembuatan grafik distribusi variabel target (`Gallstone Status`) untuk mengidentifikasi tingkat ketidakseimbangan kelas (*class imbalance*), serta distribusi fitur berbasis modalitas.
* **Matriks Korelasi Multivariat:** Pembuatan *heatmap* matriks korelasi Spearman/Pearson untuk fitur klinis utama—seperti indikator obesitas (VFR, BMI), profil lipid (TC, Triglyceride), dan enzim hati (ALT)—terhadap status batu empedu.

**Output:**
* Pemahaman mendalam terkait struktur data dan temuan awal *multicollinearity*.
* Rekomendasi fitur penting untuk diproses ke tahap pembersihan data.

---

### 2. 02_PREPROCESSING.ipynb — Data Preprocessing

Notebook ini melakukan transformasi data mentah menjadi format yang siap digunakan oleh algoritma *machine learning*, dengan fokus menjaga integritas data tanpa kebocoran (*data leakage*).

**Tahapan Utama:**
* **Pemisahan Fitur & Target:** Memisahkan matriks fitur klinis ($X$) dari target diagnosis ($y$).
* **Segmentasi Modalitas Klinis Awal:** Mengelompokkan fitur ke dalam 3 kategori medis: Fitur Demografi, Fitur Fisik (Bioimpedans), dan Fitur Kimia (Lab Darah).
* **Stratified Train-Test Split:** Membagi dataset menjadi **80% Training set** dan **20% Testing set** menggunakan metode *Stratified Sampling* (`random_state=42`) agar proporsi kelas target tetap seimbang di kedua subset.
* **Standardisasi Skala Fitur:** Menerapkan `StandardScaler` untuk menyamakan skala fitur numerik. *Scaler* hanya disesuaikan (*fit*) pada data training, kemudian mentransformasikan data training dan testing guna menghindari *data leakage*.

**Output:**
* `X_train_scaled` dan `X_test_scaled` yang mempertahankan nama kolom serta indeks asli.
* Data yang siap disegmentasikan lebih lanjut pada tahap *feature engineering*.

---

### 3. 03_FEATURE_ENGINEERING.ipynb — Feature Engineering & Modal Segmentation

Notebook ini mengimplementasikan penyaringan multikolinieritas yang ketat per modalitas serta melakukan penyimpanan objek data siap latih.

**Tahapan Utama:**
* **Analisis Korelasi per Modalitas:** Memeriksa matriks korelasi secara terisolasi pada masing-masing kelompok fitur (Fisik, Kimia, Demografi) menggunakan visualisasi *heatmap*.
* **Eliminasi Multikolinieritas:** Menerapkan fungsi kustom `drop_high_corr()` dengan ambang batas (*threshold*) korelasi tertentu (misal: `0.85`). Fitur yang redundan atau memiliki korelasi terlalu tinggi dengan fitur lain dalam modalitas yang sama akan dihapus.
* **Penyimpanan Subset Data Terstruktur:** Memisahkan dan menyimpan data hasil penyaringan ke dalam direktori khusus (`../prepare/`) dalam format `.csv` untuk fitur dan `.npy` untuk label:
  * **Fitur Fisik:** `X_train_fisik.csv`, `X_test_fisik.csv`
  * **Fitur Kimia:** `X_train_kimia.csv`, `X_test_kimia.csv`
  * **Fitur Demografi:** `X_train_demografi.csv`, `X_test_demografi.csv`
  * **Label Target:** `y_train.npy`, `y_test.npy`

**Output:**
* Subset data rendah redundansi yang siap disuplai ke masing-masing *base learner*.

---

### 4. 04_MODELING.ipynb — Multi-Modal Stacking Architecture

Notebook ini membangun arsitektur *Ensemble Stacking* dua tingkat (*Tier*) dengan memanfaatkan spesialisasi model pada tiap modalitas data.

**Tahapan Utama:**
* **Tier 1 — Pelatihan Base Learners:**
  * **Model Fisik (XGBoost):** Dilatih khusus menggunakan data fitur fisik (`X_train_fisik`) untuk menangkap pola bioimpedans tubuh.
  * **Model Kimia (LightGBM):** Dilatih khusus menggunakan data fitur kimia (`X_train_kimia`) untuk mengekstrak informasi dari hasil laboratorium darah.
* **Generasi Meta-Features (Probabilitas Kontinu):** Mengonversi hasil prediksi *Tier 1* dari data training menjadi nilai probabilitas kontinu ($P_{fisik}$ dan $P_{kimia}$).
* **Formasi Matriks Tier 2:** Menggabungkan probabilitas prediksi ($P_{fisik}$, $P_{kimia}$) dengan fitur mentah dari modalitas Demografi untuk membentuk matriks *meta-features* baru setebal 9 kolom.
* **Tier 2 — Pelatihan Meta-Learner:** Melatih model `LogisticRegression` sebagai pengambil keputusan final berdasarkan kombinasi pola prediksi *base learners* dan data demografi pasien.
* **Model Persistence:** Menyimpan objek model yang telah dilatih ke folder `../save_model/` (`model_fisik.pkl`, `model_kimia.pkl`, dan `meta_learner.pkl`) serta mengekspor hasil prediksi mentah test set ke `hasil_prediksi.csv`.

**Output:**
* Pipeline model *ensemble* yang tersimpan dan siap diuji.

---

### 5. 05_EVALUASI.ipynb — Model Evaluation & Interpretation

Notebook ini melakukan pengujian komprehensif terhadap kinerja model menggunakan data uji (*test set*) yang belum pernah dilihat sebelumnya serta memvisualisasikan matriks performa.

**Tahapan Utama:**
* **Inference Global:** Memuat kembali semua model dari direktori `../save_model/`, mengalirkan data uji melalui Tier 1 hingga menghasilkan keputusan final di Tier 2.
* **Kalkulasi Metrik Performa:** Menghitung skor evaluasi standar industri:
  * *Accuracy Score*
  * *ROC-AUC Score* (Kemampuan diskriminasi model)
  * *F1-Score* (Keseimbangan Precision dan Recall)
* **Visualisasi Evaluasi Klinis:**
  * Pembuatan *Heatmap Confusion Matrix* untuk melihat representasi *False Positive* dan *False Negative*.
  * Penggambaran Kurva ROC untuk mengevaluasi *True Positive Rate* versus *False Positive Rate*.
* **Cetak Ringkasan Box Teks:** Menampilkan teks konsol terformat rapi berupa diagram kotak ringkasan arsitektur beserta skor final metrik evaluasi.

**Output:**
* Laporan performa komprehensif beserta aset visualisasi yang disimpan ke folder `../result/`.

---

### 6. 06_FULL_PIPELINE.ipynb — End-to-End Execution

Notebook konsolidasi yang menyatukan seluruh alur kerja—mulai dari pembersihan data mentah hingga evaluasi akhir—ke dalam satu *script execution* yang runtun dan modular.

**Tahapan Utama:**
* **Kombinasi Alur Kerja Tunggal:** Mengintegrasikan fungsi *preprocessing*, pemisahan data terstratifikasi, seleksi fitur berbasis korelasi, hingga pelatihan arsitektur *stacking ensemble* dalam satu alur eksekusi tanpa jeda.
* **Skema Cross-Validation Terstratifikasi:** Menggunakan teknik seperti *Stratified K-Fold Cross Validation* untuk menghasilkan *meta-features* yang kuat dan bebas dari bias *overfitting*.
* **Analisis Koefisien Meta-Learner:** Mengekstrak nilai bobot/koefisien dari `LogisticRegression` di Tier 2 untuk mengetahui kontribusi relatif dari $P_{fisik}$, $P_{kimia}$, maupun komponen demografi.
* **Visualisasi Kontribusi Fitur:** Pembuatan grafik batang horizontal untuk koefisien *meta-learner* (diwarnai Merah untuk fitur penambah risiko, dan Biru untuk fitur penurun risiko batu empedu) serta mengekspor hasilnya menjadi `meta_learner_coefficients.png`.

**Output:**
* Pipeline produksi *end-to-end* yang dapat direplikasi penuh (*fully reproducible*) untuk kebutuhan *deployment* atau pengujian dataset baru.

**Arsitektur Model:**

```
                    [Fitur Fisik]      [Fitur Kimia]      [Fitur Demografis]
                           |                   |                    |
                    [XGBoost Base]      [LightGBM Base]   [Raw Features]
                           |                   |                    |
                     [P_fisik]          [P_kimia]          [Demografis]
                           |                   |                    |
                           +------- [Meta-Feature Matrix] -------+
                                              |
                                   [Logistic Regression]
                                     [Meta-Learner]
                                              |
                                     [Final Prediction]
```

**Output:**
- Trained base learners (XGBoost, LightGBM)
- Trained meta-learner (Logistic Regression)
- Performance metrics dan evaluation plots
- Decision boundary visualization

---

## Alur Eksekusi

**Urutan Eksekusi Pipeline:**

1. **Jalankan 01_EDA.ipynb**
   - Tujuan: Memahami data dan mengidentifikasi patterns
   - Output: Insights untuk preprocessing strategy

2. **Jalankan 02_PREPROCESSING.ipynb**
   - Tujuan: Persiapan data untuk modeling
   - Output: Standardized training/testing sets per modalitas
   - Prerequisit: Raw dataset (../data/dataset-uci.csv)

3. **Jalankan 03_FEATURE_ENGINEERING.ipynb**
   - Tujuan: Feature engineering
   - Output: Model simpan semua fitur yg diperlukan untuk train
   - Prerequisit: Menggunakan transformasi dari notebook sebelumnya

4. **Jalankan 04_Modeling.ipynb**
   - Tujuan: Membangun model dan melakukan train ensemble model
   - Output: save_model.pkl, menyimpan hasil terbaik dari train model
   - prerequisit: Menggunakan transformasi dari notebook sebelumnya

5. **Jalankan 05_Evaluasi.ipynb**
   - Tujuan: Melakukan evaluasi dari hasil train model yang sudah dilakukan
   - Output: hasil analisis akhir mengenai model yang telah digunakan
   - prerequisit: Menggunakan transformasi dari notebook sebelumnya

**Catatan Penting:**
- Setiap notebook berdiri independen namun mengikuti urutan logis
- Jalur dataset: ../data/dataset-uci.csv (relative path dari working directory)
- Semua notebooks menggunakan random_state=42 untuk reproducibility

---

## Persyaratan Sistem

### Library Utama

**Data Processing:**
- pandas >= 1.0.0
- numpy >= 1.18.0

**Machine Learning & Modeling:**
- scikit-learn >= 0.24.0
- xgboost >= 1.0.0
- lightgbm >= 2.3.0

**Visualization:**
- matplotlib >= 3.0.0
- seaborn >= 0.10.0

**Utilities:**
- warnings (standard library)

### Hardware Minimum
- RAM: 4GB
- Storage: 500MB
- Processor: Multi-core modern CPU

### Versioning
- Python >= 3.8
- Jupyter Notebook atau JupyterLab untuk eksekusi interactive

---

## Instalasi dan Penggunaan

### 1. Setup Environment

```bash
# Clone atau download repository
cd gallstone-prediction

# Buat virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
```

### 3. Persiapkan Dataset

```bash
# Pastikan struktur folder:
# gallstone-prediction/
# ├── data/
# │   └── dataset-uci.csv
# ├── 01_EDA.ipynb
# ├── 02_PREPROCESSING.ipynb
# ├── 03_FEATURE_ENGINEERING.ipynb
# └── README.md
```

### 4. Jalankan Pipeline

```bash
# Mulai Jupyter server
jupyter notebook

# Buka dan jalankan notebooks dalam urutan:
# 1. 01_EDA.ipynb
# 2. 02_PREPROCESSING.ipynb
# 3. 03_FEATURE_ENGINEERING.ipynb
```

### 5. Interpretasi Output

- **EDA Output:** Visualisasi dan statistics untuk pemahaman data
- **Preprocessing Output:** Standardized datasets per modalitas
- **Feature Engineering Output:** Model performance metrics dan evaluation plots

---

## Catatan Teknis

### Data Leakage Prevention
- StandardScaler di-fit hanya pada training set
- Test set di-transform menggunakan parameter training set
- Stratified split mempertahankan class distribution

### Model Stability
- Penggunaan random_state=42 untuk reproducibility
- Cross-validation dapat diterapkan untuk robust evaluation
- Hyperparameter values sudah dioptimasi melalui grid search

### Multimodal Integration
- Setiap modalitas diproses oleh specialized learner
- Meta-learner mengintegrasikan informasi dari base learners
- Pendekatan ini meningkatkan robustness dan generalization

---

## Lisensi dan Sumber Data

Dataset berasal dari UCI Machine Learning Repository. Penggunaan sesuai dengan lisensi dataset original.

---

**Dibuat untuk: Medical Diagnosis Prediction System**

Terakhir diupdate: 24 Juni 2026
