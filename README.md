# Predictive Maintenance Copilot – ML Models

Predictive maintenance for industrial machines: failure prediction (LSTM), failure risk scoring (XGBoost) and remaining useful life estimation (Weibull and Cox survival models).

Proyek capstone kelompok **ASAH Machine Learning Track (Dicoding Indonesia x Accenture)**. Repo ini berisi **bagian pemodelan machine learning**. Dashboard web (React + Node.js) dikerjakan anggota tim lain di repo terpisah.

## Dataset

[Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) (Kaggle): telemetry per jam (`volt`, `rotate`, `pressure`, `vibration`), catatan error, maintenance, dan failure dari 100 mesin sepanjang 2015.

Tabel telemetry, error, maintenance, dan failure digabung menjadi satu `datasets.csv` (±877 ribu baris, ±113 MB). File ini **tidak disertakan** karena melebihi batas ukuran file GitHub. Langkah penggabungannya ada di bagian *Data Preparation* pada notebook.

## Pipeline

Semua langkah ada di [`notebooks/Capstone_Project_Azure.ipynb`](notebooks/Capstone_Project_Azure.ipynb).

1. **EDA dan pre-processing**: distribusi failure per komponen dan mesin, pengurutan waktu, interpolasi missing value, fitur waktu.
2. **LSTM**: model sekuens telemetry (LSTM → Dropout → Dense dengan layer embedding 16 dimensi). Output embedding dipakai sebagai fitur tambahan (`lstm_feat`).
3. **Feature engineering**: lag (1, 3, 6 jam), rolling mean/std/max/min (3, 6, 12 jam), dan delta per mesin.
4. **XGBoost**: klasifikasi failure dengan `scale_pos_weight` untuk data sangat tidak seimbang. Split train/test berdasarkan waktu (cutoff 2015-08-01).
5. **Survival analysis**: Weibull dan Cox Proportional Hazards pada data harian + skor LSTM untuk estimasi RUL (concordance index CoxPH ≈ 0,74).
6. **Deployment**: LSTM diekspor ke TensorFlow.js, XGBoost dan model survival disajikan lewat FastAPI (`/predict_rul`).

| Komponen | Lokasi | Format |
|---|---|---|
| LSTM Failure Prediction | Browser | TFJS |
| XGBoost Risk Scoring | Backend (FastAPI) | `.pkl` |
| CoxPH + Weibull RUL | Backend | `.pkl` |
| Dashboard | Frontend React | REST + TFJS |

## Struktur

```
notebooks/   notebook lengkap (EDA → model → deployment)
models/      lstm_model.h5, embedding_model.h5, xgb.pkl, weibull_rul.pkl, coxph_rul.pkl.gz
models/lstm_tfjs/   model LSTM untuk TensorFlow.js
```

## Cara menjalankan

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Capstone_Project_Azure.ipynb
```

Sediakan `datasets.csv` (lihat bagian Dataset) di folder kerja sebelum menjalankan sel data.

> `coxph_rul.pkl.gz` dikompres agar muat diunggah; ekstrak dulu dengan `gunzip -k models/coxph_rul.pkl.gz` sebelum dipakai notebook.

> File `.pkl` hanya boleh dimuat dari sumber tepercaya (pickle dapat mengeksekusi kode saat di-load).

## Catatan dan keterbatasan

- Evaluasi XGBoost di notebook (ROC-AUC ≈ 0,9999) **terlalu optimistis**. Fitur `comp` (komponen yang sedang dirawat) berkorelasi langsung dengan label failure: hampir semua baris failure punya komponen maintenance tercatat, sedangkan di baris non-failure hanya ±0,2%. Perbaikan berikutnya: keluarkan fitur tersebut dari input dan evaluasi ulang.
- Akurasi LSTM (±99,9%) tidak informatif karena kelas failure sangat jarang; gunakan precision/recall atau PR-AUC.
- Dataset bersifat sintetis (simulasi), sehingga hasil tidak bisa langsung dianggap mewakili mesin nyata.
