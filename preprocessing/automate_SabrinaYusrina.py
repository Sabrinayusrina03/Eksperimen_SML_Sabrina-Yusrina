import pandas as pd
import numpy as np
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# KONFIGURASI FILE PATH
INPUT_FILE_PATH = 'dataset_raw/laptops.csv'
OUTPUT_DIR = 'preprocessing'
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'laptop_clean.csv')
TARGET_COLUMN = 'Price_euros'

# CLEANING DATA DASAR
def clean_data_and_prepare_for_transform(df):
    """
    Fungsi untuk membersihkan data dasar (drop kolom, hapus duplikat, cleaning teks).
    """
    df_cleaned = df.copy()

    print("--- Cleaning Data Dasar ---")
    
    # Menghapus Kolom Tidak Relevan
    df_cleaned.drop(columns=['laptop_ID'], inplace=True, errors='ignore')
    print("Kolom 'laptop_ID' dihapus.")
    
    # Menghapus Data Duplikat
    initial_rows = df_cleaned.shape[0]
    df_cleaned.drop_duplicates(inplace=True)
    rows_after_dedup = df_cleaned.shape[0]
    print(f"Duplikat dihapus: {initial_rows - rows_after_dedup} baris.")
    
    # Data Cleaning Fitur Numerik Berbentuk Teks
    # cleaning kolom ram
    if 'Ram' in df_cleaned.columns:
        df_cleaned['Ram'] = df_cleaned['Ram'].str.replace('GB', '', regex=False).astype(int)
        print("Kolom 'Ram' dibersihkan dan dikonversi ke integer.")
    
    # cleaning kolom weight
    if 'Weight' in df_cleaned.columns:
        df_cleaned['Weight'] = df_cleaned['Weight'].str.replace('kg', '', regex=False).astype(float)
        print("Kolom 'Weight' dibersihkan dan dikonversi ke float.")

    return df_cleaned

# RUNNING SKRIP
def main():
    try:
        # 1. Baca Data Mentah
        print(f"\nMembaca data dari: {INPUT_FILE_PATH}")
        df = pd.read_csv(INPUT_FILE_PATH, encoding='latin-1')

        # 2. Cleaning Dasar
        df_cleaned = clean_data_and_prepare_for_transform(df)
        print(f"Bentuk data setelah cleaning: {df_cleaned.shape}")

        # 3. Menentukan Fitur dan Target
        X = df_cleaned.drop(columns=[TARGET_COLUMN])
        y = df_cleaned[TARGET_COLUMN].reset_index(drop=True)

        # 4. Encoding Data Kategorikal & Scaling Numerik
        # Identifikasi kolom setelah cleaning
        categorical_features = X.select_dtypes(include='object').columns
        numeric_features = X.select_dtypes(include=np.number).columns
        
        print("\n--- Setup Pipeline ---")
        print(f"Fitur Numerik: {list(numeric_features)}")
        print(f"Fitur Kategorikal: {list(categorical_features)}")

        # Pipeline Processing
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        # 5. Transformasi Data
        print("Menerapkan transformasi (Scaling dan Encoding)...")
        X_processed_array = preprocessor.fit_transform(X)
        
        # 6. Konversi kembali ke DataFrame
        ohe_features = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
        
        # Gabungkan semua nama kolom
        all_cols = list(numeric_features) + list(ohe_features)
        
        df_processed = pd.DataFrame(X_processed_array, columns=all_cols)
        
        # 7. Gabungkan kembali dengan Kolom Target
        df_processed[TARGET_COLUMN] = y
        
        # 8. Menyimpan dataset hasil preprocessing
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        df_processed.to_csv(OUTPUT_FILE_PATH, index=False)
        
        print(f"\nPipeline preprocessing otomatis sukses!")
        print(f"Data siap latih disimpan di: {OUTPUT_FILE_PATH}")
        print(f"Bentuk data hasil preprocessing: {df_processed.shape}")

    except FileNotFoundError as e:
        print(f"❌ ERROR FILE NOT FOUND: Pastikan file data mentah ada di repositori: {e}")
        sys.exit(1) # Keluar dengan kode error 1
    except Exception as e:
        print(f"❌ FATAL ERROR IN PYTHON SCRIPT: {e}")
        # Hentikan workflow dan cetak error detail
        raise e

if __name__ == "__main__":
    main()