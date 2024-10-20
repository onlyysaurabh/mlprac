import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Hard-coded sample dataset with CustomerID, Annual Income (k$), Spending Score (1-100), Gender, and Age
data = {
    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Annual Income (k$)": [15, 16, 17, 55, 58, 60, 120, 125, 130, 135],  # Annual Income
    "Spending Score (1-100)": [39, 42, 45, 70, 72, 74, 20, 22, 25, 30],    # Spending Score
    "Gender": ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],  # Gender
    "Age": [22, 23, 25, 35, 36, 40, 50, 55, 60, 65]  # Age
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)
print(df.head())
print("Dataset Shape : ", df.shape)

# Extract relevant columns for clustering (Annual Income and Spending Score)
x = df.loc[:, ["Annual Income (k$)", "Spending Score (1-100)"]].values
print(x.shape)

# Fit NearestNeighbors model to find nearest neighbors
neighb = NearestNeighbors(n_neighbors=2)
nbrs = neighb.fit(x)
distances, indices = nbrs.kneighbors(x)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]

# Plot distances to determine optimal eps for DBSCAN
plt.rcParams["figure.figsize"] = (5, 3)
plt.plot(distances)
plt.title('K-distance Plot')
plt.xlabel('Points')
plt.ylabel('Distance to 2nd Nearest Neighbor')
plt.show()

# Apply DBSCAN for clustering
dbscan = DBSCAN(eps=10, min_samples=2).fit(x)
labels = dbscan.labels_

# Visualize the clusters with color coding
plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="plasma")
plt.title('DBSCAN Clustering')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# Optionally, you can analyze and display the Gender and Age distribution in each cluster:
df['Cluster'] = labels
print("\nCluster Summary:")
print(df.groupby(['Cluster']).agg({
    'Gender': 'first',  # Just showing the first Gender in each cluster for simplicity
    'Age': 'mean',  # Average Age in each cluster
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean'
}))
