
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

RANDOM_STATE = 42

DATA_PATH = "data/raw/ai4i2020.csv"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ==========================================
# 3. REMOVE UNNECESSARY COLUMNS
# ==========================================

drop_columns = [
    "UDI",
    "Product ID",

    # Target leakage prevention
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

df = df.drop(columns=drop_columns)

print("\nRemoved Unnecessary Columns")


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Machine failure"])

y = df["Machine failure"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ==========================================
# 5. IDENTIFY COLUMN TYPES
# ==========================================

categorical_columns = ["Type"]

numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

print("\nCategorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)


# ==========================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

# First: 70% training, 30% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

# Second: 15% validation, 15% testing

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("\n========== DATA SPLIT ==========")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ==========================================
# 7. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",
            StandardScaler(),
            numerical_columns
        ),

        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )

    ]

)


# ==========================================
# 8. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

}


# ==========================================
# 9. MODEL TRAINING AND VALIDATION
# ==========================================

trained_models = {}

results = []

for name, model in models.items():

    print(f"\nTraining: {name}")

    pipeline = Pipeline([

        ("preprocessing", preprocessor),

        ("model", model)

    ])

    # Training

    pipeline.fit(X_train, y_train)

    # Validation prediction

    val_probabilities = pipeline.predict_proba(X_val)[:, 1]

    # Default threshold = 0.5

    val_predictions = (val_probabilities >= 0.5).astype(int)

    # Metrics

    accuracy = accuracy_score(y_val, val_predictions)

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    pr_auc = average_precision_score(
        y_val,
        val_probabilities
    )

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc

    })

    trained_models[name] = pipeline

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(roc_auc, 4))
    print("PR AUC:", round(pr_auc, 4))


# ==========================================
# 10. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="PR AUC",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

results_df.to_csv(
    f"{OUTPUT_DIR}/model_comparison.csv",
    index=False
)


# ==========================================
# 11. SELECT BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)


# ==========================================
# 12. THRESHOLD OPTIMIZATION
# ==========================================

val_probabilities = best_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.01)

best_threshold = 0.50
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


print("\n========== OPTIMAL THRESHOLD ==========")

print("Threshold:", round(best_threshold, 2))
print("Validation F2 Score:", round(best_f2, 4))


# ==========================================
# 13. FINAL TEST EVALUATION
# ==========================================

test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


print("\n========== FINAL TEST RESULTS ==========")

print(
    "Accuracy:",
    accuracy_score(y_test, test_predictions)
)

print(
    "Precision:",
    precision_score(y_test, test_predictions, zero_division=0)
)

print(
    "Recall:",
    recall_score(y_test, test_predictions, zero_division=0)
)

print(
    "F1 Score:",
    f1_score(y_test, test_predictions, zero_division=0)
)

print(
    "F2 Score:",
    fbeta_score(y_test, test_predictions, beta=2, zero_division=0)
)

print(
    "ROC AUC:",
    roc_auc_score(y_test, test_probabilities)
)

print(
    "PR AUC:",
    average_precision_score(y_test, test_probabilities)
)


# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, test_predictions)

print("\n========== CONFUSION MATRIX ==========")

print(cm)

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# ==========================================
# 15. SAVE BEST MODEL
# ==========================================

model_artifact = {

    "pipeline": best_model,

    "threshold": float(best_threshold),

    "model_name": best_model_name,

    "feature_columns": list(X.columns)

}

joblib.dump(
    model_artifact,
    f"{MODEL_DIR}/best_model.joblib"
)


# ==========================================
# 16. SAVE TEST RESULTS
# ==========================================

test_results = {

    "model": best_model_name,

    "threshold": float(best_threshold),

    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),

    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),

    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),

    "f1_score": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),

    "f2_score": float(
        fbeta_score(y_test, test_predictions, beta=2, zero_division=0)
    ),

    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),

    "pr_auc": float(
        average_precision_score(y_test, test_probabilities)
    )

}

with open(
    f"{OUTPUT_DIR}/test_metrics.json",
    "w"
) as file:

    json.dump(test_results, file, indent=4)


print("\n================================")
print("MODEL TRAINING COMPLETED")
print("Best model saved successfully!")
print("================================")
