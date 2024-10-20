import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for data scaling

# Fixed static dataset with two features
data = pd.DataFrame({
    'Feature 1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # First feature
    'Feature 2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Second feature
})

# Display the first few rows of the original data
print("Original Data:")
print(data)

# Standardizing the data to have mean=0 and variance=1
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)  # Fit the scaler and transform the data

# Initialize PCA to reduce data to 1 component
pca = PCA(n_components=1)

# Fit PCA on the scaled data
pca.fit(data_scaled)

# Transform the data into the principal component
new_values = pca.transform(data_scaled)

# Get singular values and eigenvector for the principal component
singular_values = pca.singular_values_  # Extract singular values
eigenvector = pca.components_  # Extract the principal component (eigenvector)

# Print results of the PCA transformation
print("\nTransformed Data (reduced to 1 component):")
print(new_values)  # Display the transformed data

print("\nSingular Values:")
print(singular_values)  # Display the singular values

print("\nEigenvector (Principal Component):")
print(eigenvector)  # Display the eigenvector corresponding to the principal component

# Plotting the original data and the transformed data
plt.figure(figsize=(10, 5))

# Plot the original scaled data
plt.subplot(1, 2, 1)  # Create a subplot for the original data
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], alpha=0.5)  # Scatter plot of the original data
plt.title('Original Data')  # Title for the original data plot
plt.xlabel('Feature 1')  # X-axis label
plt.ylabel('Feature 2')  # Y-axis label

# Plot the transformed data (PCA)
plt.subplot(1, 2, 2)  # Create a subplot for the transformed data
plt.scatter(new_values, np.zeros_like(new_values), alpha=0.5)  # Scatter plot of transformed data
plt.title('Transformed Data (PCA)')  # Title for the transformed data plot
plt.xlabel('Principal Component 1')  # X-axis label for PCA
plt.yticks([])  # Hide the y-axis ticks for the transformed data plot

# Adjust layout and show the plots
plt.tight_layout()  # Optimize the layout
plt.show()  # Display all plots
