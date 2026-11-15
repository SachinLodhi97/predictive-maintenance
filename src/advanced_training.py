
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# =====================================
# 1. LOAD DATA
# =====================================

df = pd.read_csv("data/raw/ai4i2020.csv")

X = df.drop(columns=[
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

y = df["Machine failure"]


# =====================================
# 2. SPLIT DATA
# =====================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# =====================================
# 3. PREPROCESSING
# =====================================

numerical_columns = [
    col for col in X.columns if col != "Type"
]

categorical_columns = ["Type"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])


# =====================================
# 4. CLASS IMBALANCE
# =====================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("Negative Samples:", negative_count)
print("Positive Samples:", positive_count)
print("Scale Pos Weight:", scale_pos_weight)


# =====================================
# 5. XGBOOST MODEL
# =====================================

xgb_pipeline = Pipeline([

    ("preprocessing", preprocessor),

    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))

])


# =====================================
# 6. HYPERPARAMETER TUNING
# =====================================

param_grid = {

    "model__n_estimators": [200, 300, 400, 500],

    "model__max_depth": [3, 4, 5, 6],

    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],

    "model__subsample": [0.7, 0.8, 0.9],

    "model__colsample_bytree": [0.7, 0.8, 0.9]

}

search = RandomizedSearchCV(

    estimator=xgb_pipeline,

    param_distributions=param_grid,

    n_iter=12,

    scoring="average_precision",

    cv=3,

    random_state=42,

    n_jobs=-1,

    verbose=1

)

print("\nStarting Hyperparameter Tuning...")

search.fit(X_train, y_train)

best_xgb = search.best_estimator_

print("\nBest Parameters:")
print(search.best_params_)


# =====================================
# 7. VALIDATION EVALUATION
# =====================================

val_probabilities = best_xgb.predict_proba(X_val)[:, 1]

print("\nValidation PR-AUC:",
      average_precision_score(y_val, val_probabilities))

print("Validation ROC-AUC:",
      roc_auc_score(y_val, val_probabilities))


# =====================================
# 8. THRESHOLD OPTIMIZATION
# =====================================

thresholds = np.arange(0.05, 0.96, 0.01)

best_threshold = 0.5
best_f2 = -1

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = fbeta_score(
        y_val,
        predictions,
        beta=2,
        zero_division=0
    )

    if score > best_f2:
        best_f2 = score
        best_threshold = threshold

print("\nBest Threshold:", best_threshold)
print("Best Validation F2:", best_f2)


# =====================================
# 9. FINAL TEST EVALUATION
# =====================================

test_probabilities = best_xgb.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n========== FINAL TEST RESULTS ==========")

print("Accuracy:",
      accuracy_score(y_test, test_predictions))

print("Precision:",
      precision_score(y_test, test_predictions, zero_division=0))

print("Recall:",
      recall_score(y_test, test_predictions, zero_division=0))

print("F1:",
      f1_score(y_test, test_predictions, zero_division=0))

print("F2:",
      fbeta_score(y_test, test_predictions, beta=2, zero_division=0))

print("PR-AUC:",
      average_precision_score(y_test, test_probabilities))

print("ROC-AUC:",
      roc_auc_score(y_test, test_probabilities))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(classification_report(y_test, test_predictions))


# =====================================
# 10. FEATURE IMPORTANCE
# =====================================

feature_names = (
    best_xgb
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

importance = (
    best_xgb
    .named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

}).sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance_df.to_string(index=False))

importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)


# =====================================
# 11. FEATURE IMPORTANCE GRAPH
# =====================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("outputs/feature_importance.png")

plt.show()


# =====================================
# 12. SAVE MODEL
# =====================================

artifact = {

    "pipeline": best_xgb,

    "threshold": float(best_threshold),

    "model_name": "XGBoost",

    "feature_columns": list(X.columns)

}

joblib.dump(
    artifact,
    "models/xgboost_model.joblib"
)


# =====================================
# 13. SAVE METRICS
# =====================================

metrics = {

    "model": "XGBoost",

    "threshold": float(best_threshold),

    "accuracy": float(accuracy_score(y_test, test_predictions)),

    "precision": float(precision_score(
        y_test, test_predictions, zero_division=0
    )),

    "recall": float(recall_score(
        y_test, test_predictions, zero_division=0
    )),

    "f1": float(f1_score(
        y_test, test_predictions, zero_division=0
    )),

    "f2": float(fbeta_score(
        y_test, test_predictions, beta=2, zero_division=0
    )),

    "pr_auc": float(average_precision_score(
        y_test, test_probabilities
    ))

}

with open("outputs/xgboost_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nXGBOOST TRAINING COMPLETED!")
print("Model saved successfully!")
