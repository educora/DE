# Filtering Recommendation
"""
Collaborative filtering (CF) is one of the most commonly 
used recommendation system algorithms. It generates 
personalized suggestions for users based on explicit or 
implicit behavioral patterns to form predictions.
"""

from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy

# Load the dataset (e.g., the built-in movielens-100k dataset)
# Dataset ml-100k could not be found. Do you want to download it? [Y/n] y
data = Dataset.load_builtin('ml-100k')

# Define a reader with the rating scale
reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))

# Setup the data
data = Dataset.load_from_file('C:\\Users\\BilalAhmed\\.surprise_data\\ml-100k\\ml-100k\\u.data', reader=reader)

# Split the dataset for training and testing
trainset, testset = train_test_split(data, test_size=0.25)

# Use KNN to calculate similarities between users

# cosine - 
# msd - Mean squared difference (MSD)
# pearson - Pearson Correlation Coefficient
# pearson_baseline

sim_options = {
    'name': 'pearson',
    'user_based': True  # compute similarities between users
}
algo = KNNBasic(sim_options=sim_options)

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Compute and print the Mean Squared Error
accuracy.rmse(predictions)

# Predict a rating for a single user and item.
uid = str(196)  # raw user id (as in the ratings file)
iid = str(302)  # raw item id (as in the ratings file)
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
