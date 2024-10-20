import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Load the iris dataset
iris = datasets.load_iris()

# Print dataset information
print(iris.target_names)
print(iris.feature_names)
print(iris.data[0:5])
print(iris.target)

# Create a pandas DataFrame from the iris dataset
data = pd.DataFrame({
    'sepal length': iris.data[:, 0],
    'sepal width': iris.data[:, 1],
    'petal length': iris.data[:, 2],
    'petal width': iris.data[:, 3],
    'species': iris.target
})
data.head()

# Split the data into training and testing sets
x = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y = data['species']  # Labels
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Make a prediction on new data
new_data = pd.DataFrame([[3, 5, 4, 2]], columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
ans = clf.predict(new_data)
if ans[0] == 0:
    print('setosa')
elif ans[0] == 1:
    print('versicolor')
else:
    print('virginica')