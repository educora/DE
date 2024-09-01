
import pandas as pd
import numpy as np

def deduplicate_data(df, subset=None, keep='first'):
    """
    Removes duplicate rows from the DataFrame based on a subset of columns.

    Parameters:
    - df (DataFrame): The DataFrame to deduplicate.
    - subset (list of str, optional): List of column names to consider for identifying duplicates. If None, all columns are used.
    - keep (str): Determines which duplicates (if any) to keep. 
      'first' : Drop duplicates except for the first occurrence.
      'last' : Drop duplicates except for the last occurrence.
      False : Drop all duplicates.

    Returns:
    - DataFrame: A deduplicated DataFrame.
    """
    if subset is not None:
        duplicates = df[df.duplicated(subset=subset, keep=False)]
    else:
        duplicates = df[df.duplicated(keep=False)]
    
    print("Duplicate records based on the specified columns:")
    print(duplicates)

    # Remove duplicates
    return df.drop_duplicates(subset=subset, keep=keep)

# Specify the columns you believe uniquely identify a record, for example:
unique_columns = ['employee_id', 'last_name', 'first_name', 'birth_date']

# Load data with errors and identify issues from disk or from our git
# https://github.com/educora/DE/blob/main/DataSets/data%20engineering/lecture%202/errors_employee_data.csv
file_path = 'errors_employee_data.csv'  # Replace with the actual path to your file
df = pd.read_csv(file_path)

# Deduplicate the DataFrame
deduped_df = deduplicate_data(df, subset=unique_columns, keep='first')

# Save the deduplicated DataFrame
deduped_file_path = 'deduped_employee_data.csv'
deduped_df.to_csv(deduped_file_path, index=False)

print("Deduplication complete, file saved.")