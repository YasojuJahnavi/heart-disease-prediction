"""Heart Disease Prediction Using Machine Learning

Educational project using the UCI/Kaggle-style heart.csv dataset.
This model is for learning purposes only and is not a medical diagnostic tool.
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42

# Paths work when the script is run from the project root or from src/.
BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "heart.csv").exists() and (BASE_DIR.parent / "data" / "heart.csv").exists():
    DATA_PATH = BASE_DIR.parent / "data" / "heart.csv"
elif (BASE_DIR / "data" / "heart.csv").exists():
    DATA_PATH = BASE_DIR / "data" / "heart.csv"
else:
    DATA_PATH = BASE_DIR / "heart.csv"

OUTPUT_DIR = BASE_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def save_plot(filename: str) -> None:
    """Tidy, save, and close the current Matplotlib figure."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved plot: {OUTPUT_DIR / filename}")


def print_metrics(name: str, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Print classification metrics and return them for the comparison table."""
    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }
    print(f"\n========== {name.upper()} ==========")
    for metric, value in results.items():
        if metric != "Model":
            print(f"{metric:10s}: {value:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["No Heart Disease", "Heart Disease"],
        zero_division=0,
    ))
    return results


# 1. Load and inspect data
print("========== 1. LOAD DATASET ==========")
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find heart.csv. Place it at {BASE_DIR / 'heart.csv'} "
        f"or {BASE_DIR / 'data' / 'heart.csv'}."
    )

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded from: {DATA_PATH}")
print("Original shape:", df.shape)
print("\nFirst five rows:\n", df.head())
print("\nDataset information:")
df.info()
print("\nStatistical summary:\n", df.describe())
print("\nMissing values per column:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

if "target" not in df.columns:
    raise ValueError("The dataset must contain a 'target' column.")
if df.isnull().any().any():
    raise ValueError("Missing values were found. Handle them before training this script.")

df = df.drop_duplicates().reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)
print("Target distribution:\n", df["target"].value_counts().sort_index())

# 2. Exploratory Data Analysis
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="target")
plt.title("Heart Disease Target Distribution")
plt.xlabel("Target (0 = No, 1 = Yes)")
plt.ylabel("Number of records")
save_plot("target_distribution.png")

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
save_plot("correlation_heatmap.png")

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", hue="target", bins=20, kde=True, multiple="layer")
plt.title("Age Distribution by Target")
plt.xlabel("Age")
plt.ylabel("Count")
save_plot("age_distribution.png")

if "sex" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="sex", hue="target")
    plt.title("Target Distribution by Sex")
    plt.xlabel("Sex (dataset coding)")
    plt.ylabel("Count")
    save_plot("sex_distribution.png")

if "cp" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="cp", hue="target")
    plt.title("Target Distribution by Chest Pain Type")
    plt.xlabel("Chest pain type (dataset coding)")
    plt.ylabel("Count")
    save_plot("chest_pain_distribution.png")

if "thalach" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="target", y="thalach")
    plt.title("Maximum Heart Rate by Target")
    plt.xlabel("Target (0 = No, 1 = Yes)")
    plt.ylabel("Maximum heart rate")
    save_plot("heart_rate_distribution.png")

if "chol" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="target", y="chol")
    plt.title("Cholesterol by Target")
    plt.xlabel("Target (0 = No, 1 = Yes)")
    plt.ylabel("Cholesterol")
    save_plot("cholesterol_distribution.png")

# 3. Prepare features and hold out the test set.
X = df.drop(columns="target")
y = df["target"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("\n========== 3. TRAIN/TEST SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# 4. Models. Scaling is kept inside a pipeline to prevent CV data leakage.
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)),
])
tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5)

logistic_pipeline.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

logistic_pred = logistic_pipeline.predict(X_test)
logistic_prob = logistic_pipeline.predict_proba(X_test)[:, 1]
tree_pred = tree_model.predict(X_test)
tree_prob = tree_model.predict_proba(X_test)[:, 1]

logistic_results = print_metrics("Logistic Regression", y_test, logistic_pred, logistic_prob)
tree_results = print_metrics("Decision Tree (initial)", y_test, tree_pred, tree_prob)

# 5. Confusion matrices
for name, actual, predicted, filename, cmap in [
    ("Logistic Regression", y_test, logistic_pred, "confusion_matrix_logistic.png", "Blues"),
    ("Decision Tree", y_test, tree_pred, "confusion_matrix_tree.png", "Greens"),
]:
    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=["No Heart Disease", "Heart Disease"],
                yticklabels=["No Heart Disease", "Heart Disease"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    save_plot(filename)

# 6. ROC curve on the held-out test set
fpr_lr, tpr_lr, _ = roc_curve(y_test, logistic_prob)
fpr_dt, tpr_dt, _ = roc_curve(y_test, tree_prob)
auc_lr = roc_auc_score(y_test, logistic_prob)
auc_dt = roc_auc_score(y_test, tree_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.2f})")
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {auc_dt:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Test Set")
plt.legend()
save_plot("roc_curve.png")

# 7. Cross-validation on training data only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

print("\n========== 7. FIVE-FOLD CROSS-VALIDATION (TRAINING DATA) ==========")
cv_results = {}
for name, estimator in [
    ("Logistic Regression", logistic_pipeline),
    ("Decision Tree (depth=5)", tree_model),
]:
    scores = cross_validate(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results[name] = {metric: scores[f"test_{metric}"].mean() for metric in scoring}
    print(f"\n{name}")
    for metric, mean_score in cv_results[name].items():
        print(f"Mean {metric.replace('_', ' ').title():12s}: {mean_score:.4f}")

# 8. Tune Decision Tree on training data only, leaving test set untouched.
print("\n========== 8. DECISION TREE HYPERPARAMETER TUNING ==========")
param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 6],
}
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best training CV ROC-AUC:", round(grid_search.best_score_, 4))

tuned_tree = grid_search.best_estimator_
tuned_tree_pred = tuned_tree.predict(X_test)
tuned_tree_prob = tuned_tree.predict_proba(X_test)[:, 1]
tuned_tree_results = print_metrics("Tuned Decision Tree", y_test, tuned_tree_pred, tuned_tree_prob)

# 9. Compare models on the same held-out test set.
comparison = pd.DataFrame([logistic_results, tree_results, tuned_tree_results])
print("\n========== 9. HELD-OUT TEST SET COMPARISON ==========")
print(comparison.round(4).to_string(index=False))
comparison.to_csv(BASE_DIR / "model_comparison.csv", index=False)
print("Saved model comparison to:", BASE_DIR / "model_comparison.csv")

# 10. Logistic Regression coefficient magnitude (standardized inputs).
logistic_fitted_model = logistic_pipeline.named_steps["model"]
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": logistic_fitted_model.coef_[0],
    "Absolute Importance": np.abs(logistic_fitted_model.coef_[0]),
}).sort_values("Absolute Importance", ascending=False)
print("\n========== 10. LOGISTIC REGRESSION COEFFICIENTS ==========")
print(feature_importance.round(4).to_string(index=False))
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x="Absolute Importance", y="Feature")
plt.title("Logistic Regression Feature Importance")
plt.xlabel("Absolute coefficient magnitude")
plt.ylabel("Feature")
save_plot("feature_importance.png")
feature_importance.to_csv(BASE_DIR / "feature_importance.csv", index=False)

# 11. Example prediction. These are model outputs, not a clinical assessment.
patient = pd.DataFrame([{
    "age": 55,
    "sex": 1,
    "cp": 1,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2,
}])[X.columns]
patient_prediction = int(logistic_pipeline.predict(patient)[0])
patient_probability = float(logistic_pipeline.predict_proba(patient)[0, 1])
print("\n========== 11. SAMPLE PREDICTION (ILLUSTRATIVE ONLY) ==========")
print("Model output:", "Heart Disease" if patient_prediction == 1 else "No Heart Disease")
print(f"Model-estimated positive-class score: {patient_probability:.2%}")
print("This educational model is not a diagnosis and must not be used for medical decisions.")

# 12. Save reusable fitted models and the feature order.
joblib.dump(logistic_pipeline, MODEL_DIR / "logistic_regression_pipeline.joblib")
joblib.dump(tuned_tree, MODEL_DIR / "tuned_decision_tree.joblib")
print("\nSaved models in:", MODEL_DIR)
print("\nProject run completed successfully.")
print("Visualizations folder:", OUTPUT_DIR)
