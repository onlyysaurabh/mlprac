import pandas as pd  # Import pandas for data manipulation
from sklearn import datasets  # Import datasets from scikit-learn
from sklearn.model_selection import train_test_split  # For splitting the data
from sklearn.ensemble import RandomForestClassifier  # Import the Random Forest Classifier
from sklearn import metrics  # For evaluation metrics

# Load the iris dataset from scikit-learn
iris = datasets.load_iris()

# Print the names of the target classes (species)
print(iris.target_names)

# Print the names of the features in the dataset
print(iris.feature_names)

# Print the first 5 samples of the dataset (feature values)
print(iris.data[0:5])

# Print the target values (species labels) for the dataset
print(iris.target)

# Create a pandas DataFrame from the iris dataset
data = pd.DataFrame({
    'sepal length': iris.data[:, 0],  # First feature
    'sepal width': iris.data[:, 1],   # Second feature
    'petal length': iris.data[:, 2],  # Third feature
    'petal width': iris.data[:, 3],   # Fourth feature
    'species': iris.target             # Target variable (species)
})

# Display the first few rows of the DataFrame
data.head()

# Split the data into features (X) and labels (y)
x = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y = data['species']  # Labels

# Split the dataset into training and testing sets (70% training, 30% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Create a Random Forest Classifier with 100 trees
clf = RandomForestClassifier(n_estimators=100)

# Fit the classifier to the training data
clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Evaluate the model's accuracy on the test set
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Create a DataFrame for a new sample to make a prediction
new_data = pd.DataFrame([[3, 5, 4, 2]], columns=['sepal length', 'sepal width', 'petal length', 'petal width'])

# Predict the species of the new sample
ans = clf.predict(new_data)

# Print the species based on the predicted label
if ans[0] == 0:
    print('setosa')  # If the prediction is 0, it's setosa
elif ans[0] == 1:
    print('versicolor')  # If the prediction is 1, it's versicolor
else:
    print('virginica')  # If the prediction is 2, it's virginica
