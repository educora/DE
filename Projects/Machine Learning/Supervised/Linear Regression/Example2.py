"""
In this example, we use historical population data to predict future 
growth rates. Leveraging linear regression a fundamental machine
learning technique, we calculate year-over-year growth and forecast the rate,
demonstrating the power of machine learning in demographic analysis.
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Selecting Brazil's data for the analysis
country_data = data[data['country'] == 'Brazil']

# Extracting the year and population columns for Brazil
years_country = np.array([1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022, 2023]).reshape(-1, 1)
population_country = country_data[['1970 population', '1980 population', '1990 population',
                                   '2000 population', '2010 population', '2015 population',
                                   '2020 population', '2022 population', '2023 population']].values.reshape(-1, 1)

# Calculating year-over-year growth rates
growth_rates = np.diff(population_country, axis=0) / population_country[:-1] * 100
years_for_growth = years_country[1:]  # Starting from 1980 as the first growth rate is from 1970 to 1980

# Creating and training the Linear regression model for growth rates
model_growth = LinearRegression()
model_growth.fit(years_for_growth, growth_rates)

# Predicting the growth rate for the year 2024
predicted_growth_rate = model_growth.predict(np.array([[2024]]))

print(predicted_growth_rate[0][0])