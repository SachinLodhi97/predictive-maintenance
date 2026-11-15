
import pandas as pd
import numpy as np

# =====================================
# 1. Load Dataset
# =====================================

df = pd.read_csv("data/raw/ai4i2020.csv")

# =====================================
# 2. Basic Dataset Information
# =====================================

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFO ==========")
df.info()

# =====================================
# 3. Missing Values
# =====================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# =====================================
# 4. Duplicate Records
# =====================================

print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())

# =====================================
# 5. Statistical Analysis
# =====================================

print("\n========== STATISTICS ==========")
print(df.describe())

# =====================================
# 6. Target Distribution
# =====================================

print("\n========== FAILURE DISTRIBUTION ==========")
print(df["Machine failure"].value_counts())

print("\n========== FAILURE PERCENTAGE ==========")
print(
    df["Machine failure"]
    .value_counts(normalize=True) * 100
)

# =====================================
# 7. Failure Modes
# =====================================

failure_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

print("\n========== FAILURE TYPES ==========")
print(df[failure_columns].sum())

print("\nEDA COMPLETED!")
