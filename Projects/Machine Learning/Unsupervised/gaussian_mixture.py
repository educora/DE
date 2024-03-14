import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(0)

# Generating a sample dataset with a million customers
num_customers = 500
data = {
    'CustomerID': np.arange(1, num_customers + 1),
    'Annual Income (k$)': np.random.randint(15, 100, num_customers), # Random incomes between 15k and 100k
    'Spending Score (1-100)': np.random.randint(1, 101, num_customers), # Random scores between 1 and 100
    'Age': np.random.randint(18, 70, num_customers), # Random ages between 18 and 70
    'Membership Duration (years)': np.random.randint(1, 21, num_customers) # Random membership duration between 1 and 20 years
}

df_sample = pd.DataFrame(data)
df = df_sample

# Assuming df is the dataframe we previously created
features = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age', 'Membership Duration (years)']]

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Applying Gaussian Mixture Model (GMM) for clustering
gmm = GaussianMixture(n_components=4, random_state=42)
gmm_labels = gmm.fit_predict(scaled_features)

# Add the GMM cluster labels to the dataframe
df['GMM_Cluster'] = gmm_labels

# Count the number of customers in each GMM cluster
gmm_cluster_counts = df['GMM_Cluster'].value_counts().sort_index()

# Visualizing the number of customers in each GMM cluster
colors = ['red', 'blue', 'green', 'cyan'] # Re-define colors for plotting

plt.figure(figsize=(10, 6))
plt.bar(gmm_cluster_counts.index, gmm_cluster_counts.values, color=[colors[i] for i in gmm_cluster_counts.index])
plt.title('Number of Customers in Each GMM Cluster')
plt.xlabel('GMM Cluster')
plt.ylabel('Number of Customers')
plt.xticks(gmm_cluster_counts.index)
plt.show()