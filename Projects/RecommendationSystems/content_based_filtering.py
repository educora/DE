# Content Based Filtering
"""
Content-based filtering is an information retrieval method that uses 
item features to select and return items relevant to a user's query. 
This method often takes features of other items in which a user expresses 
interest into account.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample data: movie titles and their descriptions
data = {
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
    'description': [
        'An action film featuring a spy who saves the world.',
        'A romantic comedy set in New York.',
        'A documentary about the history of aviation.',
        'An action film about a secret agent on a mission.'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)
# Feature extraction: Transform text to feature vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on cosine similarity
def get_recommendations(title):
    # Get the index of the movie that matches the title
    idx = df.index[df['title'] == title][0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:3]  # Get the top 2 recommendations

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 2 most similar movies
    return df['title'].iloc[movie_indices]

# Test the system
print(get_recommendations('Movie A'))
