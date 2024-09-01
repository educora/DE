import pandas as pd
import numpy as np

# Generate a full data quality report of dataframe
"""
Steps
    Summary Statistics:- Provides a quick overview of numeric data distributions.
    Missing Values:- Counts and locations of missing data points.
    Duplicate Records:- Identification of exact and potential duplicates.
    Unique Values Count:- Helps understand cardinality in columns.
    Data Type Validation:- Verifies that data types are consistent with expectations.
    Outliers:- Detection of potential outliers in numerical data.
""" 

def data_quality_report(df):
    # Basic DataFrame info
    print("Basic DataFrame Information:\n")
    print(df.info())
    
    # Summary statistics for numerical columns
    print("\nSummary Statistics for Numerical Columns:\n")
    print(df.describe())
    
    # Missing values report
    missing_values_count = df.isna().sum()
    total_cells = np.product(df.shape)
    total_missing = missing_values_count.sum()
    print("\nMissing Values Report:\n")
    print(missing_values_count)
    print(f"\nTotal missing values: {total_missing}")
    print(f"Percentage of data that is missing: {total_missing / total_cells * 100:.2f}%")
    
    # Duplicate records report
    print("\nDuplicate Records Report:\n")
    print(f"Duplicate rows (excluding first occurrence): {df.duplicated().sum()}")
    print(f"Duplicate rows (including all occurrences): {df[df.duplicated(keep=False)].shape[0]}")
    
    # Unique values report
    print("\nUnique Values Report:\n")
    for column in df.columns:
        print(f"{column}: {df[column].nunique()} unique values")
    
    # Data types validation
    print("\nData Types in DataFrame:\n")
    print(df.dtypes)

    # Outliers detection report for numerical data (using IQR method)
    print("\nOutliers Report:\n")
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        print(f"{column}: {outliers.shape[0]} potential outliers")

# Load data with errors and identify issues from disk or from our git
# https://github.com/educora/DE/blob/main/DataSets/data%20engineering/lecture%202/errors_employee_data.csv

file_path = 'errors_employee_data.csv'  # Replace with the actual path to your file
employee_data = pd.read_csv(file_path)

# Generate data quality report
data_quality_report(employee_data)