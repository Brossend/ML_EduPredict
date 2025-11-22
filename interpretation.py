# ============================================
# ИНТЕРПРЕТАЦИЯ
# ============================================
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ============================================
# Создаём папку для графиков
# ============================================
import os
os.makedirs("plots", exist_ok=True)

# ============================================
# Загрузка модели и данных
# ============================================
model = joblib.load("models/final_model.pkl")  # или final_optuna_model.pkl
train = pd.read_csv("train.csv")

# ============================================
# Добавляем новые фичи
# ============================================
def add_features(df):
    df = df.copy()
    df['success_rate_1sem'] = df['Curricular units 1st sem (approved)'] / (df['Curricular units 1st sem (enrolled)'] + 1)
    df['success_rate_2sem'] = df['Curricular units 2nd sem (approved)'] / (df['Curricular units 2nd sem (enrolled)'] + 1)
    df['avg_success_rate'] = (df['success_rate_1sem'] + df['success_rate_2sem']) / 2
    df['total_approved'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
    df['grade_diff'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']
    return df

train = add_features(train)
X = train.drop(["id", "Target"], axis=1)
y_str = train["Target"]

# ============================================
# Кодируем таргет
# ============================================
le = LabelEncoder()
y = le.fit_transform(y_str)

print("Данные и модель загружены, фичи добавлены")

# ================================
# 1. CatBoost Feature Importance
# ================================
importances = model.get_feature_importance(type="FeatureImportance")
imp_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=imp_df.head(15), x='importance', y='feature', palette="viridis")
plt.title("Top-15 признаков — CatBoost Feature Importance")
plt.xlabel("Importance (%)")
plt.tight_layout()
plt.savefig("plots/feature_importance_catboost.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nТОП-10 признаков по CatBoost:")
print(imp_df.head(10)[['feature', 'importance']])

# ================================
# 2. Permutation Importance
# ================================
print("\nСчитаем Permutation Importance (на 10k примерах)...")

sample_idx = X.sample(n=10000, random_state=42).index
X_sample = X.loc[sample_idx]
y_sample = y[sample_idx]

perm_result = permutation_importance(
    model, X_sample, y_sample,
    n_repeats=5,
    random_state=42,
    scoring='f1_macro',
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature': X.columns,
    'importance': perm_result.importances_mean
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=perm_df.head(15), x='importance', y='feature', palette="magma")
plt.title("Top-15 признаков — Permutation Importance")
plt.xlabel("Mean decrease in F1-macro")
plt.tight_layout()
plt.savefig("plots/feature_importance_permutation.png", dpi=300, bbox_inches="tight")
plt.show()

print("ТОП-10 по Permutation Importance:")
print(perm_df.head(10))

# ================================
# 3. SHAP
# ================================
print("\nСчитаем SHAP values (может занять 1–2 минуты)...")
explainer = shap.Explainer(model)
shap_values = explainer(X.sample(5000, random_state=42))

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X.sample(5000, random_state=42), show=False)
plt.savefig("plots/shap_summary.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nГотово! Три графика сохранены в папке plots/")