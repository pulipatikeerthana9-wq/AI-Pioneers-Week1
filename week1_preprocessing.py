import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

INPUT_FILE = "data/student_performance.csv"
OUTPUT_FILE = "output/student_performance_processed.csv"


# Create output folder if it does not exist
os.makedirs("output", exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("=" * 60)
print("AI PIONEERS INTERNSHIP - WEEK 1")
print("Python for Machine Learning & Data Preprocessing")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\n1. DATASET LOADED")
print("-" * 60)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())


# --------------------------------------------------
# 3. Initial information
# --------------------------------------------------

print("\n2. DATASET INFORMATION")
print("-" * 60)

print("\nData types:")
print(df.dtypes)

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# 4. Check missing values
# --------------------------------------------------

print("\n3. MISSING VALUES BEFORE CLEANING")
print("-" * 60)

missing_before = df.isnull().sum()

print(missing_before)

print("\nTotal missing values:",
      missing_before.sum())


# --------------------------------------------------
# 5. Handle missing numerical values
# --------------------------------------------------

numerical_columns = [
    "Study_Hours",
    "Attendance",
    "Previous_Score"
]

for column in numerical_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )


# --------------------------------------------------
# 6. Handle missing categorical values
# --------------------------------------------------

categorical_columns = [
    "Gender",
    "Internet_Access",
    "Extracurricular"
]

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


print("\n4. MISSING VALUES AFTER IMPUTATION")
print("-" * 60)

missing_after = df.isnull().sum()

print(missing_after)

print("\nTotal missing values:",
      missing_after.sum())


# --------------------------------------------------
# 7. Check duplicate records
# --------------------------------------------------

print("\n5. DUPLICATE RECORDS")
print("-" * 60)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

df = df.drop_duplicates()

print("Rows after removing duplicates:",
      df.shape[0])


# --------------------------------------------------
# 8. Clean categorical values
# --------------------------------------------------

print("\n6. CATEGORICAL DATA CLEANING")
print("-" * 60)

for column in categorical_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.title()
    )

print("Categorical values cleaned successfully.")


# --------------------------------------------------
# 9. Validate numerical values
# --------------------------------------------------

print("\n7. NUMERICAL DATA VALIDATION")
print("-" * 60)

print("\nMinimum values:")
print(df[numerical_columns].min())

print("\nMaximum values:")
print(df[numerical_columns].max())


# --------------------------------------------------
# 10. Feature selection
# --------------------------------------------------

print("\n8. FEATURE SELECTION")
print("-" * 60)

print("Original columns:")
print(list(df.columns))

# Student_ID is an identifier and is not useful
# for machine learning prediction.

df = df.drop(columns=["Student_ID"])

print("\nStudent_ID removed.")

print("\nRemaining columns:")
print(list(df.columns))


# --------------------------------------------------
# 11. Encode categorical variables
# --------------------------------------------------

print("\n9. CATEGORICAL ENCODING")
print("-" * 60)

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    dtype=int
)

print("Categorical variables converted using One-Hot Encoding.")

print("\nColumns after encoding:")
print(list(df.columns))


# --------------------------------------------------
# 12. Feature normalization
# --------------------------------------------------

print("\n10. FEATURE NORMALIZATION")
print("-" * 60)

numerical_features = [
    "Age",
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Family_Income"
]

scaler = MinMaxScaler()

df[numerical_features] = scaler.fit_transform(
    df[numerical_features]
)

print("Numerical features normalized using Min-Max Scaling.")

print("\nSample normalized values:")
print(df[numerical_features].head())


# --------------------------------------------------
# 13. Exploratory Data Analysis
# --------------------------------------------------

print("\n11. EXPLORATORY DATA ANALYSIS")
print("-" * 60)

print("\nDescriptive statistics:")
print(df.describe())

print("\nCorrelation with Final_Score:")

correlation = df.corr(numeric_only=True)["Final_Score"]

print(correlation.sort_values(ascending=False))


# --------------------------------------------------
# 14. Basic statistical analysis
# --------------------------------------------------

print("\n12. FINAL SCORE ANALYSIS")
print("-" * 60)

print("Mean Final Score:",
      np.mean(df["Final_Score"]))

print("Median Final Score:",
      np.median(df["Final_Score"]))

print("Minimum Final Score:",
      np.min(df["Final_Score"]))

print("Maximum Final Score:",
      np.max(df["Final_Score"]))


# --------------------------------------------------
# 15. Save processed dataset
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n13. PROCESSED DATASET SAVED")
print("-" * 60)

print("File:", OUTPUT_FILE)
print("Final rows:", df.shape[0])
print("Final columns:", df.shape[1])


# --------------------------------------------------
# 16. Final verification
# --------------------------------------------------

print("\n14. FINAL VERIFICATION")
print("-" * 60)

print("Missing values:",
      df.isnull().sum().sum())

print("Duplicate rows:",
      df.duplicated().sum())

print("\nProcessing completed successfully!")

print("=" * 60)