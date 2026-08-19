# ============================================================
# PHẦN 1: LÀM SẠCH - TIỀN XỬ LÝ - MÃ HÓA DỮ LIỆU
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# 1. CẤU HÌNH
# ------------------------------------------------------------

# Tên dataset gốc
INPUT_FILE = "dataset.csv"

# Dataset sau khi xử lý
OUTPUT_FILE = "processed_dataset.csv"


# ------------------------------------------------------------
# 2. ĐỌC DATASET GỐC
# ------------------------------------------------------------

print("=" * 60)
print("BƯỚC 1: ĐỌC DATASET GỐC")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nKích thước dữ liệu ban đầu:")
print(df.shape)

print("\n5 dòng đầu tiên:")
print(df.head())

print("\nThông tin dữ liệu:")
df.info()


# ------------------------------------------------------------
# 3. LÀM SẠCH DỮ LIỆU
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BƯỚC 2: LÀM SẠCH DỮ LIỆU")
print("=" * 60)

# Xóa khoảng trắng ở tên cột
df.columns = df.columns.str.strip()

# Xóa các dòng bị trùng
duplicates = df.duplicated().sum()

print(f"Số dòng bị trùng: {duplicates}")

df = df.drop_duplicates()


# ------------------------------------------------------------
# 4. XỬ LÝ GIÁ TRỊ THIẾU
# ------------------------------------------------------------

print("\nSố giá trị thiếu trước khi xử lý:")
print(df.isnull().sum())

for column in df.columns:

    # Nếu là dữ liệu số
    if pd.api.types.is_numeric_dtype(df[column]):

        # Điền giá trị thiếu bằng Median
        df[column] = df[column].fillna(df[column].median())

    else:

        # Nếu là dữ liệu chữ
        mode_value = df[column].mode()

        if not mode_value.empty:
            df[column] = df[column].fillna(mode_value[0])
        else:
            df[column] = df[column].fillna("Unknown")


print("\nSố giá trị thiếu sau khi xử lý:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 5. MÃ HÓA DỮ LIỆU CHỮ THÀNH SỐ
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BƯỚC 3: MÃ HÓA DỮ LIỆU CHỮ THÀNH SỐ")
print("=" * 60)

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns

print("\nCác cột dạng chữ/categorical:")
print(list(categorical_columns))


for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

    print(f"Đã mã hóa cột: {column}")


# ------------------------------------------------------------
# 6. KIỂM TRA DỮ LIỆU SAU MÃ HÓA
# ------------------------------------------------------------

print("\nKiểu dữ liệu sau khi mã hóa:")
print(df.dtypes)

print("\n5 dòng dữ liệu sau xử lý:")
print(df.head())


# ------------------------------------------------------------
# 7. LƯU DATASET MỚI
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BƯỚC 4: LƯU DATASET MỚI")
print("=" * 60)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nĐã lưu dataset mới thành công: {OUTPUT_FILE}")

print("Kích thước dataset sau xử lý:")
print(df.shape)

print("\nHOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU!")
