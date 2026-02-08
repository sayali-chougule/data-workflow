# ============================================================
# Graduate Enrollment Pipeline
# Dataset: apps.csv
# Target: Matriculated
# ============================================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIGURATION
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

RAW_DATA_PATH = Path(".. /data/raw/apps.csv")
PROCESSED_DIR = Path(".. /data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# STEP 1: LOAD DATA
# -----------------------------
print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

apps_df = pd.read_csv(RAW_DATA_PATH)
original_rows = len(apps_df)

print(f"Loaded apps.csv with shape: {apps_df.shape}")

# -----------------------------
# STEP 2: BASIC VALIDATION
# -----------------------------
print("\n" + "=" * 70)
print("STEP 2: DATA VALIDATION")
print("=" * 70)

required_columns = ["Matriculated", "Round"]

# -----------------------------
# STEP 3: LIGHT CLEANING
# -----------------------------
print("\n" + "=" * 70)
print("STEP 3: DATA CLEANING")
print("=" * 70)

# 3.1 Remove rows with missing target
missing_target = apps_df["Matriculated"].isna().sum()
apps_df = apps_df[apps_df["Matriculated"].notna()].copy()

print(f"Removed {missing_target} rows with missing Matriculated")

# Ensure target is integer
apps_df["Matriculated"] = apps_df["Matriculated"].astype(int)

if "Banner Id" in apps_df.columns:
    apps_df["Banner Id"] = apps_df["Banner Id"].astype(str)

# -----------------------------
# Fix numeric columns stored as object (Parquet-safe)
# -----------------------------
object_cols = apps_df.select_dtypes(include="object").columns

for col in object_cols:
    # Skip obvious ID / categorical columns
    if col in ["Banner Id", "Round"]:
        continue

    # Try converting to numeric
    converted = pd.to_numeric(apps_df[col], errors="coerce")

    # If conversion makes sense (some numeric values exist), keep it
    if converted.notna().sum() > 0:
        apps_df[col] = converted



# 3.2 Remove rows with missing Essay_Length (if column exists)
if "Essay_Length" in apps_df.columns:
    missing_essay = apps_df["Essay_Length"].isna().sum()
    apps_df = apps_df[apps_df["Essay_Length"].notna()].copy()
    print(f"Removed {missing_essay} rows with missing Essay_Length")
else:
    print("Essay_Length column not found — skipping this step")

cleaned_rows = len(apps_df)

print(f"\nRows before cleaning: {original_rows}")
print(f"Rows after cleaning:  {cleaned_rows}")
print(f"Total rows removed:   {original_rows - cleaned_rows}")

# -----------------------------
# STEP 4: SAVE CLEAN DATASET
# -----------------------------
print("\n" + "=" * 70)
print("STEP 4: SAVING CLEAN DATA")
print("=" * 70)

clean_path = PROCESSED_DIR / "apps_clean.parquet"
apps_df.to_parquet(clean_path, index=False)

print(f"Saved cleaned dataset → {clean_path}")

# -----------------------------
# STEP 5: DEFINE COHORT SPLITS
# -----------------------------
print("\n" + "=" * 70)
print("STEP 5: DEFINING COHORTS")
print("=" * 70)

prediction_2026 = apps_df[apps_df["Round"].str.startswith("2026-27")].copy()
validation_2025 = apps_df[apps_df["Round"].str.startswith("2025-26")].copy()

train_test_pool = apps_df[
    ~apps_df["Round"].str.startswith("2026-27") &
    ~apps_df["Round"].str.startswith("2025-26")
].copy()

print(f"Prediction cohort (2026-27): {len(prediction_2026)}")
print(f"Validation cohort (2025-26): {len(validation_2025)}")
print(f"Train/Test pool:             {len(train_test_pool)}")
print(f"Matriculation rate (pool):   {train_test_pool['Matriculated'].mean():.2%}")

# -----------------------------
# STEP 6: TRAIN / TEST SPLIT
# -----------------------------
print("\n" + "=" * 70)
print("STEP 6: TRAIN / TEST SPLIT")
print("=" * 70)

train_df, test_df = train_test_split(
    train_test_pool,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=train_test_pool["Matriculated"]
)

print(f"Train size: {len(train_df)} ({len(train_df)/len(train_test_pool):.1%})")
print(f"  Matriculation rate: {train_df['Matriculated'].mean():.2%}")

print(f"Test size:  {len(test_df)} ({len(test_df)/len(train_test_pool):.1%})")
print(f"  Matriculation rate: {test_df['Matriculated'].mean():.2%}")

# -----------------------------
# STEP 7: SAVE SPLITS
# -----------------------------
print("\n" + "=" * 70)
print("STEP 7: SAVING DATA SPLITS")
print("=" * 70)


train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
validation_2025.to_parquet(PROCESSED_DIR / "validation_2025.parquet", index=False)
prediction_2026.to_parquet(PROCESSED_DIR / "prediction_2026.parquet", index=False)

print("Saved:")
print(" - train.parquet")
print(" - test.parquet")
print(" - validation_2025.parquet")
print(" - prediction_2026.parquet")

# -----------------------------
# FINAL PIPELINE SUMMARY
# -----------------------------
print("\n" + "=" * 70)
print("PIPELINE COMPLETE    ")
print("=" * 70)

print(f"Final cleaned rows: {cleaned_rows}")
print(f"Train/Test rows:   {len(train_df)} / {len(test_df)}")
print(f"Validation rows:   {len(validation_2025)}")
print(f"Prediction rows:   {len(prediction_2026)}")

print("\nYou can now run EDA using ONLY the processed files.")
