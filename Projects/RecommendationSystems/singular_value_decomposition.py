# Singular Value Decomposition
"""
Singular Value Decomposition (SVD) is a widely used 
technique to decompose a matrix into several component 
matrices, exposing many of the useful and interesting 
properties of the original matrix.
"""

# Rating data
# https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?resource=download&select=rating.csv

from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import Dataset, Reader

import os

# Data preperation
# Path to the dataset file
file_path = os.path.expanduser('C:\\temp\\data\\rating\\rating.csv')

# Define the format
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

# Load the data from the file
data = Dataset.load_from_file(file_path, reader=reader)


# Split the data into training and test set
trainset, testset = train_test_split(data, test_size=0.25)

# Create the SVD algorithm
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Compute and print the Mean Squared Error
accuracy.rmse(predictions)

# Assuming the user ID and item ID are known
userid = '196'
movieid = '302'
rating = 4

# Predict rating
prediction = algo.predict(userid, movieid, r_ui=rating, verbose=True)

print(prediction)
