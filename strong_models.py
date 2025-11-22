# ============================================
# ПРОКАЧАННЫЕ МОДЕЛИ
# ============================================

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from catboost import CatBoostClassifier

from data import X_train, y_train, X_val, y_val

models = {}
results = {}

# ============================================
# 1. RandomForest
# ============================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)

models["RandomForest"] = rf
results["RandomForest"] = {
    "Accuracy": accuracy_score(y_val, rf_pred),
    "F1_macro": f1_score(y_val, rf_pred, average="macro")
}

print("=== RandomForest ===")
print(results["RandomForest"])
print(classification_report(y_val, rf_pred))


# ============================================
# 2. GradientBoosting
# ============================================

gb = GradientBoostingClassifier(
    learning_rate=0.05,
    n_estimators=300,
    max_depth=3
)

gb.fit(X_train, y_train)
gb_pred = gb.predict(X_val)

models["GradientBoosting"] = gb
results["GradientBoosting"] = {
    "Accuracy": accuracy_score(y_val, gb_pred),
    "F1_macro": f1_score(y_val, gb_pred, average="macro")
}

print("\n=== GradientBoosting ===")
print(results["GradientBoosting"])
print(classification_report(y_val, gb_pred))


# ============================================
# 3. CatBoost — ЛУЧШЕЕ РЕШЕНИЕ
# ============================================

cat = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='MultiClass',
    verbose=0,
    random_seed=42
)

cat.fit(X_train, y_train)
cat_pred = cat.predict(X_val)

models["CatBoost"] = cat
results["CatBoost"] = {
    "Accuracy": accuracy_score(y_val, cat_pred),
    "F1_macro": f1_score(y_val, cat_pred, average="macro")
}

print("\n=== CatBoost ===")
print(results["CatBoost"])
print(classification_report(y_val, cat_pred))


# ============================================
# Итоговая таблица
# ============================================

import pandas as pd
print("\n=== Сравнение моделей ===")
df_results = pd.DataFrame(results).T
print(df_results)
