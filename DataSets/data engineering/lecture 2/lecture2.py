import pandas as pd
import random
import numpy as np

# Introduce missing values randomly in the DataFrame
def introduce_missing_values(df, percentage=20):
    # percentage: percentage of total data to be set as NaN
    n_rows, n_cols = df.shape
    n_missing = int(n_rows * n_cols * percentage / 100)
    for _ in range(n_missing):
        i = random.randint(0, n_rows - 1)
        j = random.randint(0, n_cols - 1)
        df.iat[i, j] = np.nan

# Introduce spelling inconsistencies
def introduce_spelling_errors(df, column_name, percentage=5):
    # Only apply to columns specified in column_name if they are string types
    if column_name in df.columns and df[column_name].dtype == object:
        n_rows = len(df)
        indices = random.sample(range(n_rows), k=int(n_rows * percentage / 100))
        for i in indices:
            original_text = str(df.at[i, column_name])
            if len(original_text) > 1:  # ensure there's something to swap
                char_pos = random.randint(0, len(original_text) - 2)
                # Swap two consecutive characters
                new_text = (original_text[:char_pos] + original_text[char_pos + 1] + 
                            original_text[char_pos] + original_text[char_pos + 2:])
                df.at[i, column_name] = new_text

# Alter phone number formats randomly
def randomize_phone_formats(df, column_name):
    if column_name in df.columns and df[column_name].dtype == object:
        n_rows = len(df)
        for i in range(n_rows):
            phone = str(df.at[i, column_name])
            if '-' in phone:
                new_format = phone.replace('-', '').replace('x', ' ext ')
                df.at[i, column_name] = new_format        

# Function to introduce duplicates into the DataFrame
def introduce_duplicates(df, percentage=10):
    n_rows = len(df)
    n_duplicates = int(n_rows * percentage / 100)
    duplicate_indices = random.sample(list(df.index), k=n_duplicates)
    duplicates = df.loc[duplicate_indices].copy()
    return pd.concat([df, duplicates]).sample(frac=1).reset_index(drop=True)  # Shuffle and reset index

# load employess data
file_path = 'C:\\temp\\data\\data engineering\\lecture 2\\employee_data.csv'  # Replace with the actual path to your file
employee_data = pd.read_csv(file_path)

# Applying the functions to the dataframe
employee_data = introduce_duplicates(employee_data, percentage=10)  # Add 10% duplicates
introduce_missing_values(employee_data, percentage=10)  # 10% of data will have missing values
introduce_spelling_errors(employee_data, 'last_name', percentage=5)  # 5% chance to introduce a spelling error in last names
randomize_phone_formats(employee_data, 'home_phone')  # Change phone number formats

# Print the modified DataFrame to check the changes
print(employee_data.head())

# save file back
new_file_path = 'C:\\temp\\data\\data engineering\\lecture 2\\errors_employee_data.csv'  # Replace with the actual path to your file
employee_data.to_csv(new_file_path, index=False)