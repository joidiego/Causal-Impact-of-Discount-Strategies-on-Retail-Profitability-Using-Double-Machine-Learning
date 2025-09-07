#!/usr/bin/env python
"""
run_pipeline.py

Pipeline utama untuk proyek:
1. Load & bersihkan data
2. Agregasi & feature engineering
3. Estimasi efek kausal (Double ML)
4. Simulasi kebijakan (policy backtest)
5. Simpan hasil & visualisasi

Dijalankan sebagai:
    python run_pipeline.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pastikan folder plots ada
os.makedirs("plots", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Atur seed untuk reproducibility
np.random.seed(42)

# -------------------------------
# 1. Import Modul Lokal
# -------------------------------
try:
    from src.data.clean import load_and_clean_data
    from src.features.create_features import create_causal_features
    from src.models.double_ml import ContinuousDoubleML
    from src.evaluation.backtest import policy_backtest, plot_policy_backtest
    from src.utils.seeding import set_seed
    set_seed(42)  # Reproducibility
except ModuleNotFoundError as e:
    raise ImportError(f"Pastikan struktur folder benar: {e}")


# -------------------------------
# 2. Fungsi Utama
# -------------------------------
def main():
    print("🚀 Memulai pipeline data science...")

    # --- Langkah 1: Load & Clean Data ---
    print("📁 1. Memuat dan membersihkan data...")
    raw_path = "data/raw/Sample - Superstore.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Data tidak ditemukan di {raw_path}. Pastikan file ada di folder data/raw/")

    df = load_and_clean_data(raw_path)
    clean_path = "data/processed/superstore_clean.csv"
    df.to_csv(clean_path, index=False)
    print(f"✅ Data bersih disimpan di {clean_path}")

    # --- Langkah 2: Feature Engineering ---
    print("⚙️  2. Membuat fitur untuk model kausal...")
    try:
        X, T, Y = create_causal_features(df)
        print(f"📊 Fitur siap: X.shape={X.shape}, T.shape={T.shape}, Y.shape={Y.shape}")
    except Exception as e:
        raise RuntimeError(f"Gagal membuat fitur: {e}")

    # --- Langkah 3: Model Kausal (Double ML) ---
    print("🧠 3. Menjalankan Double Machine Learning...")
    try:
        model = ContinuousDoubleML(n_folds=5, random_state=42)
        causal_effect = model.fit(X, T, Y)
        print(f"🎯 Estimasi Causal Effect: {causal_effect:.4f}")
        
        # Simpan hasil
        with open("reports/causal_effect.txt", "w") as f:
            f.write(f"Causal Effect of Discount on Profit: {causal_effect:.4f}\n")
    except Exception as e:
        raise RuntimeError(f"Gagal menjalankan Double ML: {e}")

    # --- Langkah 4: Backtest Kebijakan ---
    print("📊 4. Melakukan backtest kebijakan diskon...")
    try:
        discount_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        backtest_results = policy_backtest(causal_effect, discount_levels, baseline_profit=df['Profit'].mean())
        plot_policy_backtest(backtest_results)
        print("✅ Backtest selesai dan grafik disimpan.")
    except Exception as e:
        print(f"⚠️  Backtest gagal: {e}")

    # --- Langkah 5: Ringkasan Hasil ---
    print("\n" + "="*50)
    print("✅ PIPELINE SELESAI")
    print("="*50)
    print(f"📌 Causal Effect: {causal_effect:.4f}")
    print(f"💡 Interpretasi: Setiap kenaikan diskon 1 poin → profit berubah ${causal_effect:.2f}")
    print(f"📊 Lihat visualisasi di folder: plots/")
    print(f"📄 Laporan: reports/writeup.pdf")
    print("="*50)


# -------------------------------
# 5. Jalankan jika sebagai script
# -------------------------------
if __name__ == "__main__":
    main()