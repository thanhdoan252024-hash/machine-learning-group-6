# ============================================================
# PHẦN 2: ĐỌC DATASET ĐÃ XỬ LÝ VÀ CHIA TRAIN / TEST 80-20
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. ĐỌC DATASET ĐÃ XỬ LÝ
# ------------------------------------------------------------

DATA_FILE = "processed_dataset.csv"

# Cột cần dự đoán
TARGET_COLUMN = "charges"

print("=" * 60)
print("ĐỌC DATASET ĐÃ XỬ LÝ")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\nKích thước dataset:")
print(df.shape)

print("\n5 dòng đầu:")
print(df.head())


# ------------------------------------------------------------
# 2. TÁCH BIẾN ĐẦU VÀO X VÀ BIẾN MỤC TIÊU y
# ------------------------------------------------------------

X = df.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]

print("\nCác biến đầu vào X:")
print(X.columns.tolist())

print("\nBiến cần dự đoán y:")
print(TARGET_COLUMN)


# ------------------------------------------------------------
# 3. CHIA TRAIN / TEST THEO TỶ LỆ 80 / 20
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------------------
# 4. HIỂN THỊ KẾT QUẢ
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("KẾT QUẢ CHIA TRAIN / TEST")
print("=" * 60)

print("\nTRAIN 80%")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTEST 20%")

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ------------------------------------------------------------
# 5. LƯU CÁC TẬP TRAIN / TEST
# ------------------------------------------------------------

X_train.to_csv(
    "X_train.csv",
    index=False
)

X_test.to_csv(
    "X_test.csv",
    index=False
)

y_train.to_csv(
    "y_train.csv",
    index=False
)

y_test.to_csv(
    "y_test.csv",
    index=False
)


print("\nĐÃ LƯU THÀNH CÔNG:")

print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")

print("\nHOÀN THÀNH CHIA TRAIN / TEST 80-20!")
