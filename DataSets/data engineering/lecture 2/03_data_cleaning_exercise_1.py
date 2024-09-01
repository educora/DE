import pandas as pd
import numpy as np

# Load data with errors and identify issues from disk or from our git
# https://github.com/educora/DE/blob/main/DataSets/data%20engineering/lecture%202/errors_employee_data.csv
file_path = 'errors_employee_data.csv'  # Replace with the actual path to your file
df = pd.read_csv(file_path)

# Handling missing values
## Depending on the nature of data, you can choose to fill with default values or drop rows/columns
default_values = {
    'title_of_courtesy': 'No Title',  # Filling missing titles with a placeholder
    'home_phone': 'Unknown',  # Default for missing phone numbers
    'notes': 'No comments'  # Default for missing notes
}
df.fillna(default_values, inplace=True)

# Correcting data types
## Convert 'hire_date' and 'birth_date' from string to datetime
df['birth_date'] = pd.to_datetime(df['birth_date'], format='%d/%m/%Y', errors='coerce')
df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')

# Removing duplicates
df.drop_duplicates(inplace=True)

# Standardize text data
## Ensure text fields are clean and standardized
text_columns = ['last_name', 'first_name', 'title', 'address', 'city', 'country']
for col in text_columns:
    df[col] = df[col].str.strip().str.capitalize()

# Handling inconsistencies in 'region'
## Assuming regions should be uppercase
df['region'] = df['region'].str.upper()

# Cleaning phone numbers to have a consistent format
df['home_phone'] = df['home_phone'].replace('\D', '', regex=True).replace(r'^\s*$', np.nan, regex=True)

# Handle outliers in numerical columns such as 'extension'
# For simplicity, the 'extension' column is assumed to be numerical and the outlier treatment is basic capping
if df['extension'].dtype != np.number:
    df['extension'] = pd.to_numeric(df['extension'], errors='coerce')

Q1 = df['extension'].quantile(0.25)
Q3 = df['extension'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['extension'] = np.where(df['extension'] < lower_bound, lower_bound, df['extension'])
df['extension'] = np.where(df['extension'] > upper_bound, upper_bound, df['extension'])

# Save the cleaned data back to disk
cleaned_file_path = 'cleaned_employee_data.csv'
df.to_csv(cleaned_file_path, index=False)

print("Data cleaning completed and file saved.")
