import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Function to evaluate and plot metrics for random binary data
def evaluate_random_data(n=1000):
    # Generate random binary data for actual and predicted values
    # Here, we use a binomial distribution with n=1 (binary outcome) and p=0.9 (90% chance of 1)
    actual = np.random.binomial(1, 0.9, size=n)
    predicted = np.random.binomial(1, 0.9, size=n)

    # Calculate various performance metrics
    accuracy = metrics.accuracy_score(actual, predicted)  # Calculate accuracy
    precision = metrics.precision_score(actual, predicted)  # Calculate precision
    recall = metrics.recall_score(actual, predicted)  # Calculate recall
    specificity = metrics.recall_score(actual, predicted, pos_label=0)  # Calculate specificity
    fScore = metrics.f1_score(actual, predicted)  # Calculate F1 score
    confMat = metrics.confusion_matrix(actual, predicted)  # Generate confusion matrix

    # Print the calculated metrics
    print(f"Confusion Matrix: \n{confMat}")
    print(f"Accuracy: {round(accuracy, 3)} \tPrecision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)} \t\tSpecificity: {round(specificity, 3)}")
    print(f"F Score: {round(fScore, 3)}")

    # Display the confusion matrix using a heatmap
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confMat, display_labels=[0, 1])
    cm_display.plot(cmap=plt.cm.bone_r)

    # Add labels and title to the plot
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Random Data') 
    plt.show()

# Function to evaluate Random Forest on digits dataset
def evaluate_digits_dataset():
    # Load the digits dataset
    X, y = load_digits(return_X_y=True)

    # Split the dataset into training and testing sets
    # Here, we use a test size of 25% and a fixed random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)

    # Train the Random Forest model
    clf = RandomForestClassifier(random_state=23)  # Initialize the classifier with a fixed random state
    clf.fit(X_train, y_train)  # Fit the model on the training data

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Compute the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix using seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='g')
    plt.ylabel('Actual', fontsize=13)
    plt.xlabel('Prediction', fontsize=13)
    plt.title('Confusion Matrix for Digits Dataset', fontsize=17)
    plt.show()

    # Calculate and print accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

# Evaluate random binary data with hardcoded value of n=1000
evaluate_random_data(n=1000)

# Evaluate digits dataset
evaluate_digits_dataset()
