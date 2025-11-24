import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

RAW_DIR = "raw-data/"
PROC_DIR = "processed-data/"
os.makedirs(PROC_DIR, exist_ok=True)

# 1. Load Data ----------------------------------------------
df_train = pd.read_csv(RAW_DIR + "UNSW_NB15_training-set.csv")
df_test  = pd.read_csv(RAW_DIR + "UNSW_NB15_testing-set.csv")
df = pd.concat([df_train, df_test], ignore_index=True)

# 2. Clean Data ------------------------------------------------
# Drop metadata columns that are not useful for ML
drop_cols = ["id"] 
df = df.drop(columns=drop_cols)

# Replace missing / inf values
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# 3. Encode Labels -----------------------------------------------
# Binary: 1=attack, 0=benign
df['binary_label'] = df['label']

# Multiclass: encode attack categories
encoder = LabelEncoder()
df['multi_label'] = encoder.fit_transform(df['attack_cat'].astype(str))

# 4. Select Features ---------------------------------------------
X = df.drop(columns=['label', 'attack_cat', 'binary_label', 'multi_label'])
y_binary = df['binary_label']
y_multi  = df['multi_label']

# Ensure all features numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# 5. Split Data --------------------------------------------------
# Stratified split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_binary, test_size=0.3, random_state=37, stratify=y_binary
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=37, stratify=y_temp
)

# 6. Save Splits -------------------------------------------------
X_train.to_csv(PROC_DIR + "X_train.csv", index=False)
y_train.to_csv(PROC_DIR + "y_train.csv", index=False)
X_val.to_csv(PROC_DIR + "X_val.csv", index=False)
y_val.to_csv(PROC_DIR + "y_val.csv", index=False)
X_test.to_csv(PROC_DIR + "X_test.csv", index=False)
y_test.to_csv(PROC_DIR + "y_test.csv", index=False)

# 6. Prints -------------------------------------------------
print("Dataset shape:", df.shape)
print("Columns:", df.columns[:10], "...")

print("\nBinary distribution:\n", df['binary_label'].value_counts())
print("Classes (multiclass):", encoder.classes_)

print("\nTrain size:", X_train.shape[0])
print("Validation size:", X_val.shape[0])
print("Test size:", X_test.shape[0])
