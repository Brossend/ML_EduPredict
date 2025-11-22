# ============================================
# БАЗОВЫЕ МОДЕЛИ
# ============================================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from data import X_train, y_train, X_val, y_val

# ============================================
# Logistic Regression
# ============================================

logreg = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_val)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_val, logreg_pred))
print("F1 (macro):", f1_score(y_val, logreg_pred, average="macro"))
print("\nClassification report:")
print(classification_report(y_val, logreg_pred))


# ============================================
# Confusion matrix
# ============================================
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_val, logreg_pred), annot=True, fmt="d", cmap="Blues")
plt.title("LogReg - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ============================================
# Decision Tree Classifier
# ============================================
tree = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

tree.fit(X_train, y_train)
tree_pred = tree.predict(X_val)

print("\n=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_val, tree_pred))
print("F1 (macro):", f1_score(y_val, tree_pred, average="macro"))
print("\nClassification report:")
print(classification_report(y_val, tree_pred))

# ============================================
# Confusion matrix
# ============================================
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_val, tree_pred), annot=True, fmt="d", cmap="Greens")
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
