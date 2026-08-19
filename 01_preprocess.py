import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. ĐỌC DATASET GỐC
# ==========================================

INPUT_FILE = "insurance.csv"
OUTPUT_FILE = "processed_dataset.csv"

df = pd.read_csv(INPUT_FILE)

print("Kích thước dữ liệu ban đầu:", df.shape)
print("\n5 dòng đầu:")
print(df.head())


# ==========================================
# 2. LÀM SẠCH DỮ LIỆU
# ==========================================

# Xóa khoảng trắng trong tên cột
df.columns = df.columns.str.strip()

# Xóa dữ liệu trùng
df = df.drop_duplicates()

# Xử lý dữ liệu thiếu
for column in df.columns:

    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())

    else:
        df[column] = df[column].fillna(df[column].mode()[0])


# ==========================================
# 3. MÃ HÓA CHỮ THÀNH SỐ
# ==========================================

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

print("\nCác cột cần mã hóa:")
print(list(categorical_columns))

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

    print("Đã mã hóa:", column)


# ==========================================
# 4. LƯU DATASET MỚI
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDataset sau khi xử lý:")
print(df.head())

print("\nKích thước:", df.shape)

print(
    "\nĐã lưu thành công:",
    OUTPUT_FILE
)
