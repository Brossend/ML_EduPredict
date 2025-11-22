# ============================================
# ФИНАЛЬНЫЙ САБМИШЕН С ЛУЧШИМИ ПАРАМЕТРАМИ OPTUNA
# ============================================

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ============================================
# Загружаем лучшие параметры
# ============================================
best_params = joblib.load("models/best_params_optuna.pkl")
print("Лучшие параметры от Optuna:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ============================================
# Загружаем данные + фичи
# ============================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_ids = test["id"].copy()

train = train.drop("id", axis=1)
test = test.drop("id", axis=1)

def add_features(df):
    df = df.copy()
    df['success_rate_1sem'] = df['Curricular units 1st sem (approved)'] / (df['Curricular units 1st sem (enrolled)'] + 1)
    df['success_rate_2sem'] = df['Curricular units 2nd sem (approved)'] / (df['Curricular units 2nd sem (enrolled)'] + 1)
    df['avg_success_rate'] = (df['success_rate_1sem'] + df['success_rate_2sem']) / 2
    df['total_approved'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
    df['grade_diff'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']
    return df

train = add_features(train)
test = add_features(test)

cat_features = ['Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
                'Previous qualification', 'Nacionality', "Mother's qualification",
                "Father's qualification", "Mother's occupation", "Father's occupation",
                'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
                'Gender', 'Scholarship holder', 'International']

X_train = train.drop('Target', axis=1)
y_train = LabelEncoder().fit_transform(train['Target'])

# ============================================
# Добавляем обязательные параметры CatBoost
# ============================================
best_params.update({
    'loss_function': 'MultiClass',
    'random_seed': 42,
    'verbose': 100,
    'cat_features': cat_features,
    'od_type': 'Iter',
    'od_wait': 50
})

# ============================================
# Обучаем финальную модель на всех данных
# ============================================
from catboost import CatBoostClassifier
final_model = CatBoostClassifier(**best_params)
final_model.fit(X_train, y_train)

# ============================================
# Предсказываем и сохраняем submission
# ============================================
predictions = final_model.predict(test)
predictions_labels = LabelEncoder().fit_transform(train['Target']).classes_[predictions.flatten()]

submission = pd.DataFrame({
    'id': test_ids,
    'Target': predictions_labels
})
submission.to_csv("submission_best_optuna.csv", index=False)

print("\nsubmission_best_optuna.csv")