import pandas as pd
import numpy as np

# Hard-coded dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
})

# Function to calculate entropy
def entropy(y):
    value_counts = y.value_counts()
    probabilities = value_counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

# Function to calculate information gain
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / np.sum(counts)) * entropy(data[data[feature] == values[i]][target])
        for i in range(len(values))
    )
    return total_entropy - weighted_entropy

# ID3 algorithm implementation
def id3(data, target, features):
    unique_classes = data[target].unique()
    if len(unique_classes) == 1:
        return unique_classes[0]
    if len(features) == 0:
        return data[target].mode()[0]

    gains = [information_gain(data, feature, target) for feature in features]
    best_feature_index = np.argmax(gains)
    best_feature = features[best_feature_index]

    tree = {best_feature: {}}

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = id3(subset, target, features.drop(best_feature))
        tree[best_feature][value] = subtree

    return tree

# Function to print the tree with predicted labels
def print_tree(tree, depth=0):
    for key in tree:
        print("  " * depth + str(key))
        if isinstance(tree[key], dict):
            print_tree(tree[key], depth + 1)
        else:
            print("  " * (depth + 1) + "-> Predicted label: " + str(tree[key]))

# Function to make a prediction based on the decision tree
def predict(tree, sample):
    while isinstance(tree, dict):
        feature = next(iter(tree))
        value = sample[feature]
        tree = tree[feature].get(value, None)
        if tree is None:
            return "Unknown"
    return tree

# Define features and target
target = 'Play'
features = data.columns.drop(target)

# Build the decision tree
decision_tree = id3(data, target, features)

# Hard-coded input for prediction
input_sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Hot',
    'Humidity': 'High',
    'Wind': 'Weak'
}

# Make a prediction
predicted_label = predict(decision_tree, input_sample)

# Print the decision tree with predicted labels
print("Decision Tree:")
print_tree(decision_tree)

# Print input and predicted label
print("\nInput Sample:")
for feature, value in input_sample.items():
    print(f"{feature}: {value}")

print(f"Predicted label: {predicted_label}")
