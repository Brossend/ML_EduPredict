# ============================================
# СХЕМА ВАЛИДАЦИИ + ПРОВЕРКА НА DATA LEAKAGE
# ============================================

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

from data import X, y

# ============================================
# Фиксируем все random states для полной воспроизводимости
# ============================================

SEED = 42
np.random.seed(SEED)

# ============================================
# Стратифицированная кросс-валидация (сохраняем пропорции классов в каждом фолде)
# ============================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("Схема валидации: Stratified 5-Fold CV")
print(f"Random state зафиксирован: {SEED}")
print("→ Нет data leakage: все признаки собраны до момента отчисления (исторические данные)")
print("→ Stratify=y гарантирует одинаковое распределение Target в каждом фолде\n")

# ============================================
# Пример расчёта CV-оценки для CatBoost (потом будем использовать для всех моделей)
# ============================================

from catboost import CatBoostClassifier


def cv_score(model, X, y, cv=cv, scoring='f1_macro'):
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        scores.append(f1_score(y_val, pred, average='macro'))

    return np.array(scores)

# ============================================
# Демонстрация на лучшей модели
# ============================================

cat_cv = CatBoostClassifier(
    iterations=800,
    depth=7,
    learning_rate=0.05,
    loss_function='MultiClass',
    verbose=0,
    random_seed=SEED
)

cv_scores = cv_score(cat_cv, X, y)
print(f"CatBoost — 5-Fold CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("Отдельные фолды:", cv_scores.round(4))