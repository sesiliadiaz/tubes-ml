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
[Raw Data] → [01_EDA] → [02_PREPROCESSING] → [03_FEATURE_ENGINEERING] → [Model Output]
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

### 1. 01_EDA.ipynb - Exploratory Data Analysis

Notebook pertama melakukan eksplorasi menyeluruh terhadap dataset untuk memahami karakteristik, pola, dan hubungan antar variabel.

**Tahapan Utama:**

a) **Loading dan Inspeksi Data**
   - Membaca dataset UCI medical
   - Menampilkan dimensi dataset (jumlah baris dan kolom)
   - Menampilkan info struktur data dan tipe variabel

b) **Analisis Kualitas Data**
   - Deteksi missing values di seluruh dataset
   - Identifikasi dan penanganan duplikat records
   - Evaluasi kompletness data per fitur

c) **Statistik Deskriptif**
   - Ringkasan statistik untuk fitur medis pilihan
   - Measure of central tendency dan dispersi
   - Quartile analysis untuk outlier detection

d) **Visualisasi Distribusi**
   - Distribusi kelas target (Gallstone Status)
   - Histograms dan distribusi per modalitas fitur
   - Identifikasi imbalance dalam target variable

e) **Analisis Korelasi Multivariat**
   - Korelasi obesity indicators (VFR, BMI) terhadap status batu empedu
   - Profil lipid analysis (TC, LDL, HDL, Triglyceride) vs target
   - Matriks korelasi antar-fitur klinis utama
   - Heatmap untuk visualisasi hubungan fitur

f) **Exploratory Insights**
   - Identifikasi fitur yang paling berkorelasi dengan target
   - Deteksi potential multicollinearity issues
   - Pattern discovery dalam data demografis

**Output:**
- Pemahaman mendalam tentang struktur dan karakteristik dataset
- Identifikasi fitur-fitur penting
- Rekomendasi untuk tahap preprocessing

---

### 2. 02_PREPROCESSING.ipynb - Data Preprocessing

Notebook kedua melakukan transformasi data untuk mempersiapkan input model machine learning.

**Tahapan Utama:**

a) **Feature Selection**
   - Pemisahan fitur klinis (X) dari target diagnosis (y)
   - Organisasi fitur berdasarkan tiga modalitas:
     * Fitur demografis (7 fitur)
     * Fitur fisik (18 fitur)
     * Fitur kimia (13 fitur)

b) **Train-Test Split**
   - Pembagian dataset menjadi training (80%) dan testing (20%)
   - Implementasi stratified split untuk mempertahankan class balance
   - Menggunakan random_state=42 untuk reproducibility
   - Output: X_train (255, 38), X_test (64, 38)

c) **Standardisasi Fitur**
   - Aplikasi StandardScaler untuk normalisasi skala fitur
   - Fit scaler hanya pada training set untuk mencegah data leakage
   - Transform testing set menggunakan parameter dari training set
   - Preservasi column names dan index dalam output

d) **Segmentasi Fitur Berdasarkan Modalitas**
   - Pembentukan subset per modalitas untuk training data:
     * X_train_demografi
     * X_train_fisik
     * X_train_kimia
   - Pembentukan subset per modalitas untuk testing data:
     * X_test_demografi
     * X_test_fisik
     * X_test_kimia

**Output:**
- Standardized training dan testing sets
- Fitur-fitur yang terorganisir per modalitas
- Data siap untuk tahap feature engineering dan modeling

**Prinsip Penting:**
- Mencegah data leakage dengan fit scaler hanya pada training set
- Mempertahankan stratifikasi untuk class balance
- Modular organization untuk memfasilitasi per-modalitas processing

---

### 3. 03_FEATURE_ENGINEERING.ipynb - Feature Engineering & Stacking Ensemble

Notebook ketiga mengimplementasikan feature engineering lanjutan dan membangun arsitektur stacking ensemble multi-modal dua tingkat.

**Tahapan Utama:**

a) **Data Preparation Review**
   - Reload dataset dan preprocessing ulang
   - Replicasi split dan standardisasi dari notebook sebelumnya
   - Pemisahan fitur per modalitas untuk processing terpisah

b) **Analisis Korelasi Per Modalitas**
   - Heatmap korelasi untuk fitur fisik
   - Heatmap korelasi untuk fitur kimia
   - Heatmap korelasi untuk fitur demografis
   - Identifikasi multicollinearity patterns

c) **Eliminasi Multikolinieritas**
   - Implementasi fungsi drop_high_corr()
   - Threshold korelasi 0.85 untuk dropping highly correlated features
   - Filtering per modalitas untuk mengurangi redundansi
   - Output: feature subset yang lebih lean dengan information preservation

d) **Tier 1 - Base Learners**

   **Model 1: XGBoost (Modalitas Fisik)**
   - Hyperparameters:
     * n_estimators: 200
     * max_depth: 4
     * learning_rate: 0.05
     * subsample: 0.8
     * colsample_bytree: 0.8
   - Training pada fitur fisik (X_train_fisik)
   - Output: probabilitas kelas (P_fisik)

   **Model 2: LightGBM (Modalitas Kimia)**
   - Hyperparameters:
     * n_estimators: 200
     * max_depth: 4
     * learning_rate: 0.05
     * num_leaves: 31
     * subsample: 0.8
   - Training pada fitur kimia (X_train_kimia)
   - Output: probabilitas kelas (P_kimia)

   **Model 3: Logistic Regression (Modalitas Demografis)**
   - Direct use of demografis features sebagai complementary input
   - Dimasukkan ke Tier 2 sebagai raw features

e) **Tier 2 - Meta-Features Formation**

   Pembentukan matriks meta-features yang menggabungkan:
   - Probabilitas prediksi dari XGBoost (P_fisik)
   - Probabilitas prediksi dari LightGBM (P_kimia)
   - Raw demografis features
   - Format: meta_train = [P_fisik | P_kimia | Demografis]

f) **Tier 2 - Meta-Learner Training**

   **Meta-Classifier: Logistic Regression**
   - Hyperparameters:
     * C: 1.0
     * max_iter: 1000
     * random_state: 42
   - Training pada matriks meta-features
   - Learning dari pola kombinasi base learner predictions

g) **Model Inference dan Prediksi**
   - Inference pada test set untuk semua base learners
   - Pengumpulan probabilitas base learners
   - Formation matriks meta-features untuk test
   - Final prediction dari meta-learner

h) **Model Evaluation**

   **Metrics Perhitungan:**
   - Accuracy Score
   - ROC-AUC Score
   - F1-Score (weighted average)
   - Precision dan Recall per kelas
   - Matthews Correlation Coefficient (MCC)

   **Visualization:**
   - Confusion Matrix heatmap
   - ROC-AUC curve
   - Classification report detail

i) **Analisis Separabilitas Ruang Keputusan**
   - Scatter plot dari meta-features 2D
   - Visualisasi decision boundary
   - Color-coded points berdasarkan predicted class
   - Assessment spatial separability antar kelas

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
   - Tujuan: Feature engineering dan training stacking ensemble
   - Output: Model terlatih dan evaluation metrics
   - Prerequisit: Menggunakan transformasi dari notebook sebelumnya

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

Terakhir diupdate: 2024
