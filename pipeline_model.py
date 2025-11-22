# ============================================
# ПАЙПЛАЙН — CatBoost с DataFrame
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
import joblib
import os

SEED = 42
np.random.seed(SEED)

# -------------------------------
# Загрузка и базовая подготовка
# -------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_ids = test["id"].copy()

train = train.drop("id", axis=1)
test = test.drop("id", axis=1)

# -------------------------------
# Определяем типы колонок
# -------------------------------
cat_features = [
    'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
    'Previous qualification', 'Nacionality', "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
    'Gender', 'Scholarship holder', 'International'
]

num_features = [col for col in train.columns if col not in cat_features + ['Target']]

print(f"Категориальных признаков: {len(cat_features)}")
print(f"Числовых признаков: {len(num_features)}")

# -------------------------------
# Создаём новые признаки
# -------------------------------
def add_features(df):
    df = df.copy()
    df['success_rate_1sem'] = df['Curricular units 1st sem (approved)'] / (
                df['Curricular units 1st sem (enrolled)'] + 1)
    df['success_rate_2sem'] = df['Curricular units 2nd sem (approved)'] / (
                df['Curricular units 2nd sem (enrolled)'] + 1)
    df['avg_success_rate'] = (df['success_rate_1sem'] + df['success_rate_2sem']) / 2
    df['total_approved'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
    df['grade_diff'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']
    return df

train = add_features(train)
test = add_features(test)

# -------------------------------
# Добавляем новые числовые признаки в список
# -------------------------------
new_num_features = ['success_rate_1sem', 'success_rate_2sem', 'avg_success_rate', 'total_approved', 'grade_diff']
num_features.extend(new_num_features)

# -------------------------------
# Стандартизация числовых признаков (опционально)
# -------------------------------
scaler = StandardScaler()
train[num_features] = scaler.fit_transform(train[num_features])
test[num_features] = scaler.transform(test[num_features])

# -------------------------------
# Кодируем таргет
# -------------------------------
X = train.drop('Target', axis=1)
y = train['Target']

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# -------------------------------
# CV + обучение
# -------------------------------
print("\nЗапуск обучения с CV...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_encoded), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    model = CatBoostClassifier(
        iterations=1200,
        depth=7,
        learning_rate=0.05,
        loss_function='MultiClass',
        verbose=100,
        random_seed=SEED,
        cat_features=cat_features
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=100)
    pred = model.predict(X_val)
    score = f1_score(y_val, pred, average='macro')
    cv_scores.append(score)
    print(f"Fold {fold} F1-macro: {score:.5f}")

print(f"\nИТОГО CV F1-macro: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}")

# -------------------------------
# Финальное обучение на всех данных
# -------------------------------
print("\nОбучение финальной модели на всех данных...")
final_model = CatBoostClassifier(
    iterations=1200,
    depth=7,
    learning_rate=0.05,
    loss_function='MultiClass',
    verbose=100,
    random_seed=SEED,
    cat_features=cat_features
)

final_model.fit(X, y_encoded, verbose=100)

# -------------------------------
# Сохраняем модель и энкодер
# -------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/final_model.pkl")
joblib.dump(le_target, "models/label_encoder_target.pkl")
joblib.dump(scaler, "models/num_scaler.pkl")
print("Модель и энкодер сохранены!")

# -------------------------------
# Предсказание на тест
# -------------------------------
test_scaled = test.copy()
test_scaled[num_features] = scaler.transform(test[num_features])
predictions = final_model.predict(test_scaled)
predictions_labels = le_target.inverse_transform(predictions.ravel())

submission = pd.DataFrame({
    'id': test_ids,
    'Target': predictions_labels
})
submission.to_csv("submission.csv", index=False)
print("submission.csv готов!")
