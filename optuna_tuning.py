# ============================================
# OPTUNA - ПРОДВИНУТАЯ ГИПЕРОПТИМИЗАЦИЯ
# ============================================

import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

SEED = 42
np.random.seed(SEED)

# ============================================
# Загружаем данные с новыми фичами
# ============================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
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

X = train.drop('Target', axis=1)
y = LabelEncoder().fit_transform(train['Target'])

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 800, 2000),
        'depth': trial.suggest_int('depth', 5, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 128, 512),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'od_type': 'Iter',
        'od_wait': 50,
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1:average=Macro',
        'random_seed': SEED,
        'verbose': False,
        'cat_features': cat_features
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
        pred = model.predict(X_val)
        scores.append(f1_score(y_val, pred, average='macro'))

    return np.mean(scores)

print("Запуск Optuna (50 trials) — это займёт 15–25 минут...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nЛучший F1-macro на CV: {study.best_value:.5f}")
print("Лучшие параметры:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# ============================================
# Сохраняем лучшие параметры
# ============================================
joblib.dump(study.best_params, "models/best_params_optuna.pkl")
print("Лучшие параметры сохранены в models/best_params_optuna.pkl")

# ============================================
# Финальная модель с лучшими параметрами
# ============================================
best_params = study.best_params
best_params.update({
    'loss_function': 'MultiClass',
    'random_seed': SEED,
    'verbose': 100,
    'cat_features': cat_features
})

final_optuna_model = CatBoostClassifier(**best_params)
final_optuna_model.fit(X, y)

joblib.dump(final_optuna_model, "models/final_optuna_model.pkl")
print("Финальная модель с Optuna сохранена!")