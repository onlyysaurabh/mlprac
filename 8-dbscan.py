import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import pandas as pd  # Import pandas for data manipulation
from sklearn.neighbors import NearestNeighbors  # For finding nearest neighbors
from sklearn.cluster import DBSCAN  # Import DBSCAN for clustering

# Hard-coded sample dataset with CustomerID, Annual Income (k$), Spending Score (1-100), Gender, and Age
data = {
    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Annual Income (k$)": [15, 16, 17, 55, 58, 60, 120, 125, 130, 135],  # Annual Income
    "Spending Score (1-100)": [39, 42, 45, 70, 72, 74, 20, 22, 25, 30],    # Spending Score
    "Gender": ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],  # Gender
    "Age": [22, 23, 25, 35, 36, 40, 50, 55, 60, 65]  # Age
}

# Convert the data into a pandas DataFrame for easier manipulation
df = pd.DataFrame(data)
print(df.head())  # Display the first few rows of the DataFrame
print("Dataset Shape : ", df.shape)  # Print the shape of the DataFrame

# Extract relevant columns for clustering (Annual Income and Spending Score)
x = df.loc[:, ["Annual Income (k$)", "Spending Score (1-100)"]].values
print(x.shape)  # Print the shape of the extracted data for clustering

# Fit NearestNeighbors model to find nearest neighbors
# Setting n_neighbors=2 to find the nearest neighbor for each point
neighb = NearestNeighbors(n_neighbors=2)
nbrs = neighb.fit(x)  # Fit the model to the data
distances, indices = nbrs.kneighbors(x)  # Get distances and indices of nearest neighbors
distances = np.sort(distances, axis=0)  # Sort the distances
distances = distances[:, 1]  # Select the distances to the 2nd nearest neighbor

# Plot distances to determine optimal eps for DBSCAN
plt.rcParams["figure.figsize"] = (5, 3)  # Set figure size for the plot
plt.plot(distances)  # Plot the distances
plt.title('K-distance Plot')  # Title for the plot
plt.xlabel('Points')  # Label for the x-axis
plt.ylabel('Distance to 2nd Nearest Neighbor')  # Label for the y-axis
plt.show()  # Display the plot

# Apply DBSCAN for clustering
# - eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other
# - min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point
dbscan = DBSCAN(eps=10, min_samples=2).fit(x)  # Fit the DBSCAN model to the data
labels = dbscan.labels_  # Get the labels assigned by DBSCAN

# Visualize the clusters with color coding
plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="plasma")  # Scatter plot with clusters colored
plt.title('DBSCAN Clustering')  # Title for the plot
plt.xlabel("Annual Income (k$)")  # Label for the x-axis
plt.ylabel("Spending Score (1-100)")  # Label for the y-axis
plt.show()  # Display the plot

# Optionally, analyze and display the Gender and Age distribution in each cluster
df['Cluster'] = labels  # Add the cluster labels to the DataFrame
print("\nCluster Summary:")
# Group by cluster and aggregate gender, age, annual income, and spending score
print(df.groupby(['Cluster']).agg({
    'Gender': 'first',  # Display the first Gender in each cluster for simplicity
    'Age': 'mean',  # Calculate the average Age in each cluster
    'Annual Income (k$)': 'mean',  # Calculate the average Annual Income in each cluster
    'Spending Score (1-100)': 'mean'  # Calculate the average Spending Score in each cluster
}))
