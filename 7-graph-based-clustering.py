import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import seaborn as sns  # Import Seaborn for enhanced visualizations
import warnings  # For handling warnings
from sklearn.datasets import make_blobs  # For generating synthetic datasets
from mst_clustering import MSTClustering  # Import the MST Clustering class

# Ignore specific warnings related to elementwise operations
warnings.filterwarnings("ignore", message="elementwise")

# Function to plot the Minimum Spanning Tree (MST) and clusters
def plot_mst(model, cmap='rainbow'):
    # Extract the fitted data from the model
    X = model.X_fit_
    
    # Create subplots for the full and trimmed MST
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    
    # Loop through axes to plot both full and trimmed MST
    for axi, full_graph, colors in zip(ax, [True, False], ['lightblue', model.labels_]):
        # Get the segments of the MST
        segments = model.get_graph_segments(full_graph=full_graph)
        
        # Plot the MST segments
        axi.plot(segments[0], segments[1], '-k', zorder=1, lw=1)
        
        # Scatter plot of the data points, colored by cluster labels
        axi.scatter(X[:, 0], X[:, 1], c=colors, cmap=cmap, zorder=2)
        axi.axis('tight')  # Set axis limits to fit the data tightly
    
    # Set titles for the plots
    ax[0].set_title('Full Minimum Spanning Tree', size=16)
    ax[1].set_title('Trimmed Minimum Spanning Tree', size=16)

# Generate sample data using make_blobs
# - 200 samples, 6 centers (clusters)
X, y = make_blobs(200, centers=6)

# Initial scatter plot of the generated data
plt.scatter(X[:, 0], X[:, 1], c='lightblue')

# Apply MST Clustering
# - cutoff_scale: Controls the scale at which clusters are cut
# - approximate: Whether to use an approximate method
model = MSTClustering(cutoff_scale=2, approximate=False)
labels = model.fit_predict(X)  # Fit the model and predict cluster labels

# Scatter plot of the data points colored by the assigned cluster labels
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')

# Plot the MST for the clusters
plot_mst(model)

# Show all plots
plt.show()
