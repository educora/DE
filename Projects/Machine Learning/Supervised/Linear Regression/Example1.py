"""
This example demonstrates the application of a linear regression model to 
historical population data. The goal is to establish a relationship 
between the year and population growth rates, which is then used to predict 
the growth rate, showcasing data-driven forecasting in action.
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Selecting Brazil's data for the analysis
country_data = data[data['country'] == 'Brazil']

# Extracting the year and population columns for India
years = np.array([[1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022, 2023]]).reshape(-1, 1)
population = country_data[['1970 population', '1980 population', '1990 population', 
                           '2000 population', '2010 population', '2015 population', 
                           '2020 population', '2022 population', '2023 population']].values.reshape(-1, 1)

# Creating and training the Linear regression model
model = LinearRegression()
model.fit(years, population)

# Predicting the population for the year 2024
predicted_population = model.predict(np.array([[2050]]))

print(predicted_population[0][0])