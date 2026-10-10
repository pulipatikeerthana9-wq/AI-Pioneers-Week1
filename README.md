# AI Pioneers Internship - Week 1

## Python for Machine Learning & Data Preprocessing

This project was completed as part of the AI Pioneers Machine Learning Internship, Week 1.

### Objective

The objective of this project is to understand the fundamentals of Python-based machine learning data preprocessing. The project demonstrates how to load, inspect, clean, transform, normalize, and analyze a sample student performance dataset using Python.

### Technologies Used

* Python
* NumPy# AI Pioneers Internship - Week 1: Python for ML & Data Preprocessing

Preprocessing pipeline for a 40-record Student Performance dataset (pandas, NumPy, scikit-learn).

## What the script does
Loads the CSV, reports missing values, fills numeric gaps with the median (and categorical gaps with the mode), checks duplicates, cleans categorical text, drops `Student_ID`, one-hot encodes `Gender`, `Internet_Access` and `Extracurricular`, Min-Max scales five numeric features (target `Final_Score` is not scaled), prints descriptive statistics and correlations with `Final_Score`, and saves the result.

## Structure
- `week1_preprocessing.py` - the script
- `data/student_performance.csv` - raw data (40 rows x 10 columns, 3 missing values)
- `output/student_performance_processed.csv` - result (40 rows x 12 columns, 0 missing values)
- `requirements.txt` - dependencies

## Run
```
pip install -r requirements.txt
python week1_preprocessing.py
```
Run from the project folder (paths are relative). Expected final lines: `Missing values: 0`, `Duplicate rows: 0`, `Processing completed successfully!`.

## Note
The dataset appears synthetic (very smooth relationships, perfectly balanced classes); results must not be generalised to real students.
* Pandas
* Scikit-learn

### Dataset

The project uses a sample Student Performance Dataset containing information such as:

* Age
* Gender
* Study Hours
* Attendance
* Previous Score
* Internet Access
* Extracurricular Activities
* Family Income
* Final Score

The dataset contains 40 student records and 10 original columns.

### Preprocessing Steps

The following preprocessing operations were performed:

1. Loaded the dataset using Pandas.
2. Inspected the dataset structure and data types.
3. Identified missing values.
4. Handled missing numerical values using median imputation.
5. Checked and handled duplicate records.
6. Cleaned categorical values.
7. Validated numerical features.
8. Removed the Student_ID column because it is an identifier rather than a predictive feature.
9. Converted categorical variables using one-hot encoding.
10. Applied Min-Max normalization to selected numerical features.
11. Performed basic exploratory data analysis.
12. Saved the processed dataset as a CSV file.

### Results

The preprocessing program successfully processed all 40 records.

* Missing values before preprocessing: 3
* Missing values after preprocessing: 0
* Duplicate records: 0
* Original columns: 10
* Final columns: 12
* Minimum Final Score: 55
* Maximum Final Score: 96
* Mean Final Score: 76.375
* Median Final Score: 75.5

### Project Structure

```text
AI-Pioneers-Week1/
│
├── data/
│   └── student_performance.csv
│
├── output/
│   └── student_performance_processed.csv
│
├── week1_preprocessing.py
│
└── README.md
```

### How to Run

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn
```

Run the preprocessing program:

```bash
python week1_preprocessing.py
```

The processed dataset will be generated at:

```text
output/student_performance_processed.csv
```

### Learning Outcome

This project provided practical experience in Python-based data preprocessing and demonstrated how raw data can be transformed into a cleaner and machine-learning-ready format. It also provided hands-on practice with Pandas, NumPy, categorical encoding, missing-value handling, feature selection, normalization, and exploratory data analysis.
