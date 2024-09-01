import pandas as pd
import numpy as np

# Before moving your cleaned data to a database, there are several final checks and preparations to ensure that your data is truly ready for integration and further use. Here’s a checklist to consider:
# Load final dataset
# https://github.com/educora/DE/blob/main/DataSets/data%20engineering/lecture%202/errors_employee_data.csv
file_path = 'deduped_employee_data.csv'  # Replace with the actual path to your file
df = pd.read_csv(file_path)

# 1. Final Review of Data Types
# Ensure that each column in your DataFrame has the appropriate data type that matches the expected schema in the database. This avoids issues with type mismatches during data insertion.
# Checking data types
print(df.dtypes)

# 2. Validation Against Schema
#Validate your data against the expected database schema. This includes checking constraints like unique keys, foreign keys, and other relational integrity constraints.
# Ensure 'employee_id' is unique and not null
assert df['employee_id'].is_unique and df['employee_id'].notnull().all(), "Employee ID must be unique and not null."

# 3. Normalization Check
# If your data model involves normalization, ensure that the data is properly separated into the respective tables/entities, matching the normalized structure of your database.
# ...

# 4. Data Range and Integrity Checks
# Verify that the data falls within acceptable ranges and adheres to business logic constraints, such as valid dates, permissible values, and logical relationships between fields.
from datetime import datetime
assert df['birth_date'].max() <= datetime.today(), "Birth dates should be in the past."

# 5. Dealing with Remaining Missing Values
# If there are still missing values that might have been overlooked, decide on a strategy to handle them based on the database requirements.
# Fill remaining missing values with defaults or drop them
df.fillna(value={'some_column': 'default_value'}, inplace=True)

# 6. Performance Optimization
# Consider indexing columns that will be frequently queried to improve performance once the data is in the database.
# ...

# 7. Security and Compliance Check
# Ensure that any sensitive data is properly encrypted or anonymized, and that your data handling complies with relevant data protection regulations (e.g., GDPR, HIPAA).

# 8. Final Data Export
# Export your data to the format required for database importation. This could be CSV, JSON, SQL inserts, or direct DataFrame to SQL methods.
# Export to CSV if needed
final_path = 'final_cleaned_data.csv'
df.to_csv(final_path, index=False)

# Or, write directly to SQL database using SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine('sqlite:///path_to_your_database.db')
df.to_sql('employee_table', con=engine, index=False, if_exists='append')

# 9. Logging and Documentation
# Maintain logs of the cleaning process and transformations applied. Document the data cleaning steps and final schema for future reference and audits.
# ...

# 10. Backup the Original Data
# Always keep a backup of the original data before making the final move to the database. This helps in recovering from any unforeseen issues with the cleaned data.
# ...