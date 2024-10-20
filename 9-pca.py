import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Fixed static dataset
data = pd.DataFrame({
    'Feature 1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature 2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
})

# Display the first few rows of the data
print("Original Data:")
print(data)

# Standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Initialize PCA
pca = PCA(n_components=1)

# Fit PCA on the scaled data
pca.fit(data_scaled)

# Transform the data
new_values = pca.transform(data_scaled)

# Get singular values and eigenvector
singular_values = pca.singular_values_
eigenvector = pca.components_

# Print results
print("\nTransformed Data (reduced to 1 component):")
print(new_values)

print("\nSingular Values:")
print(singular_values)

print("\nEigenvector (Principal Component):")
print(eigenvector)

# Plotting the original data vs. transformed data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], alpha=0.5)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(new_values, np.zeros_like(new_values), alpha=0.5)
plt.title('Transformed Data (PCA)')
plt.xlabel('Principal Component 1')
plt.yticks([])  # Hides the y-axis ticks

plt.tight_layout()
plt.show()
