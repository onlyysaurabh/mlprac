import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from sklearn import datasets  # Import datasets from scikit-learn
from sklearn.model_selection import train_test_split  # For splitting the data
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.metrics import accuracy_score  # For calculating accuracy

# Generate synthetic data for classification
# - n_samples: Total number of samples
# - n_features: Total number of features
# - n_informative: Number of informative features
# - n_redundant: Number of redundant features
# - n_clusters_per_class: Number of clusters per class
# - random_state: Seed for reproducibility
X, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, 
                                    n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split the generated data into training and testing sets
# - test_size: Proportion of the dataset to include in the test split
# - random_state: Seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a Support Vector Machine (SVM) classifier
# - kernel: Specifies the kernel type ('linear' for a linear decision boundary)
# - C: Regularization parameter
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)  # Fit the model to the training data

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)  # Compare predicted labels to true labels
print(f"Accuracy: {accuracy:.2f}")  # Print the accuracy rounded to 2 decimal places

# Function to plot decision boundary
def plot_decision_boundary(X, y, model):
    # Calculate the min and max values of the features for setting up the grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid for plotting decision boundaries
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Make predictions over the grid to create the decision boundary
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # Combine the grid into a single array
    Z = Z.reshape(xx.shape)  # Reshape to match the meshgrid shape

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)  # Plot the decision boundary
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)  # Plot the samples
    plt.title("SVM Decision Boundary")  # Title of the plot
    plt.xlabel("Feature 1")  # Label for x-axis
    plt.ylabel("Feature 2")  # Label for y-axis
    plt.show()  # Display the plot

# Plot the decision boundary using the test data and the trained SVM model
plot_decision_boundary(X_test, y_test, svm_model)
