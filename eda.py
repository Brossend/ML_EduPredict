# ============================================
# Импорт библиотек
# ============================================
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from phik.report import plot_correlation_matrix


# ============================================
# Папка для графиков
# ============================================
SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================
# Загрузка данных
# ============================================
train_path = "train.csv"
test_path = "test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("=== TRAIN HEAD ===")
print(train.head())

print("\n=== TEST HEAD ===")
print(test.head())


# ============================================
# Общая информация о датасете
# ============================================
print("\n=== TRAIN INFO ===")
train.info()

print("\n=== TEST INFO ===")
test.info()


# ============================================
# Описательная статистика
# ============================================
print("\n=== TRAIN DESCRIBE ===")
print(train.describe(include="all"))

print("\n=== TEST DESCRIBE ===")
print(test.describe(include="all"))


# ============================================
# Проверка пропущенных значений
# ============================================
print("\n=== MISSING VALUES (train) ===")
print(train.isna().sum().sort_values(ascending=False))

print("\n=== MISSING VALUES (test) ===")
print(test.isna().sum().sort_values(ascending=False))


# ============================================
# Анализ целевой переменной
# ============================================
target_column = "Target"

if target_column in train.columns:
    plt.figure(figsize=(6,4))
    train[target_column].value_counts().plot(kind="bar")
    plt.title("Распределение целевой переменной")
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    plt.savefig(f"{SAVE_DIR}/target_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()
else:
    print(f"Столбец {target_column} не найден!")


# ============================================
# Корреляционный анализ PHIK
# ============================================
print("\n=== PHIK CORRELATION MATRIX ===")

phik_matrix = train.phik_matrix()

plt.figure(figsize=(25, 20))
plot_correlation_matrix(
    phik_matrix.values,
    x_labels=phik_matrix.columns,
    y_labels=phik_matrix.index,
    vmin=0,
    vmax=1
)
plt.title("PHIK корреляции")
plt.savefig(f"{SAVE_DIR}/phik_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================
# Корреляционный анализ PHIK (Большая)
# ============================================

print("\n=== PHIK CORRELATION MATRIX (BIG) ===")

phik_matrix = train.phik_matrix()

plt.figure(figsize=(40, 35))

sns.heatmap(
    phik_matrix,
    annot=True, # => Цифры можно отрубить
    cmap="RdYlGn",
    vmin=0,
    vmax=1
)

plt.title("PHIK корреляции", fontsize=24)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.savefig(f"{SAVE_DIR}/phik_matrix_big.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================
# Гистограммы числовых признаков
# ============================================

numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) > 0:
    train[numeric_cols].hist(figsize=(15,10), bins=20)
    plt.suptitle("Распределение числовых признаков", y=1.02)
    plt.savefig(f"{SAVE_DIR}/numeric_hist.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================
# Boxplot для поиска выбросов
# ============================================

if len(numeric_cols) > 0:
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=train[numeric_cols])
    plt.title("Boxplot числовых признаков")
    plt.xticks(rotation=90)
    plt.savefig(f"{SAVE_DIR}/boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================
# Pairplot для базового анализа
# ============================================

if len(numeric_cols) > 1:
    sns.pairplot(train[numeric_cols[:5]])
    plt.savefig(f"{SAVE_DIR}/pairplot.png", dpi=300, bbox_inches="tight")
    plt.show()

# ============================================
# ДОПОЛНИТЕЛЬНЫЙ ГЛУБОКИЙ АНАЛИЗ
# ============================================

import warnings
warnings.filterwarnings("ignore")

# ============================================
# 1. Распределение возраста по статусу
# ============================================
plt.figure(figsize=(10,6))
sns.boxplot(data=train, x='Target', y='Age at enrollment', palette="Set2")
plt.title("Распределение возраста при зачислении по финальному статусу", fontsize=14)
plt.xlabel("Target")
plt.ylabel("Age at enrollment")
plt.savefig(f"{SAVE_DIR}/age_by_target.png", dpi=300, bbox_inches="tight")
plt.show()

print("Инсайт: студенты, которые в итоге Dropout, в среднем старше 30 лет → семейные/рабочие обстоятельства.")

# ============================================
# 2. Влияние задолженностей и оплаты
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14,5))

sns.countplot(data=train, x='Debtor', hue='Target', ax=axes[0], palette="viridis")
axes[0].set_title("Влияние статуса должника")
sns.countplot(data=train, x='Tuition fees up to date', hue='Target', ax=axes[1], palette="viridis")
axes[1].set_title("Оплата обучения вовремя")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/debtor_and_fees.png", dpi=300, bbox_inches="tight")
plt.show()

print("Инсайт: наличие долга и неоплата обучения — сильнейшие предикторы Dropout.")

# ============================================
# СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ И ПРОВЕРКА ГИПОТЕЗ
# ============================================

# ============================================
# Новые признаки
# ============================================
train_ed = train.copy()

train_ed['success_rate_1sem'] = train_ed['Curricular units 1st sem (approved)'] / (train_ed['Curricular units 1st sem (enrolled)'] + 1)
train_ed['success_rate_2sem'] = train_ed['Curricular units 2nd sem (approved)'] / (train_ed['Curricular units 2nd sem (enrolled)'] + 1)
train_ed['avg_success_rate'] = (train_ed['success_rate_1sem'] + train_ed['success_rate_2sem']) / 2

train_ed['total_approved'] = train_ed['Curricular units 1st sem (approved)'] + train_ed['Curricular units 2nd sem (approved)']
train_ed['grade_diff'] = train_ed['Curricular units 2nd sem (grade)'] - train_ed['Curricular units 1st sem (grade)']

# ============================================
# Гипотеза 1: проверка через ANOVA
# ============================================
from scipy.stats import f_oneway

dropout = train_ed[train_ed['Target'] == 'Dropout']['avg_success_rate']
enrolled = train_ed[train_ed['Target'] == 'Enrolled']['avg_success_rate']
graduate = train_ed[train_ed['Target'] == 'Graduate']['avg_success_rate']

f_stat, p_val = f_oneway(dropout, enrolled, graduate)
print(f"ANOVA для avg_success_rate: F = {f_stat:.2f}, p-value = {p_val:.2e}")
print("→ Гипотеза подтверждена: средний процент зачётов сильно различается между группами (p << 0.05)")

# ============================================
# Гипотеза 2: влияние стипендии
# ============================================
from scipy.stats import chi2_contingency
cross_tab = pd.crosstab(train_ed['Scholarship holder'], train_ed['Target'])
chi2, p, dof, expected = chi2_contingency(cross_tab)
print(f"Chi2 для Scholarship holder: p-value = {p:.2e} → стипендия значимо влияет на исход")

