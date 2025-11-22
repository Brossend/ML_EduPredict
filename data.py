# ============================================
# ПРЕДОБРАБОТКА ДАННЫХ
# ============================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ============================================
# Сохраним id, чтобы вернуть позже для submission
# ============================================
test_ids = test["id"]

# ============================================
# 2. УДАЛЕНИЕ НЕИНФОРМАТИВНЫХ ПРИЗНАКОВ
# id = уникальный идентификатор → ненужен для обучения
# ============================================

train = train.drop(columns=["id"])
test = test.drop(columns=["id"])

# ============================================
# 3. LABEL ENCODING для категориальных фичей
# ============================================

cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
cat_cols.remove("Target")  # целевую переменную кодируем отдельно

label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# ============================================
# Кодирование TARGET
# ============================================
le_target = LabelEncoder()
train["Target"] = le_target.fit_transform(train["Target"])

# ============================================
# 4. МАСШТАБИРОВАНИЕ ЧИСЛОВЫХ ФИЧЕЙ
# ============================================

num_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols.remove("Target")  # не масштабируем целевую переменную

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

# ============================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================

X = train.drop(columns=["Target"])
y = train["Target"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # чтобы баланс классов сохранялся
)

print("Размер обучающей выборки:", X_train.shape)
print("Размер валидационной выборки:", X_val.shape)
print("Количество классов Target:", y_train.nunique())
