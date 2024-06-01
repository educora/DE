# Cosine Similarity
""" 
Cosine similarity measures the similarity between two vectors 
of an inner product space. It is measured by the cosine of the 
angle between two vectors and determines whether two vectors are 
pointing in roughly the same direction.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Sample data: movie titles, genre vectors, and average ratings
data = {
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
    'action': [1, 0, 0, 1],
    'comedy': [0, 1, 0, 0],
    'documentary': [0, 0, 1, 0],
    'average_rating': [4.5, 4.0, 4.5, 3.5]  # Assume 5 as highest
}
# Create a DataFrame
df = pd.DataFrame(data)
# Scale the 'average_rating' to normalize its influence
scaler = StandardScaler()
df['scaled_rating'] = scaler.fit_transform(df[['average_rating']])
# Create a feature matrix for similarity calculation, including genres and scaled rating
feature_matrix = df[['action', 'comedy', 'documentary', 'scaled_rating']]
# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(feature_matrix)
# Function to get recommendations based on cosine similarity
def get_recommendations(title):
    # Get the index of the movie that matches the title
    idx = df.index[df['title'] == title][0]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the top recommendations, excluding the first (itself)
    sim_scores = sim_scores[1:3]  # Get the top 2 recommendations
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 2 most similar movies
    return df['title'].iloc[movie_indices]
# Test the system
print(get_recommendations('Movie A'))

