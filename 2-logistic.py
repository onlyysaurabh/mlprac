import matplotlib.pyplot as plt
import numpy as np

# Function to get user input for data
def get_user_input():
    x1 = list(map(float, input("Enter annual incomes (Lakhs) separated by spaces: ").split()))
    x2 = list(map(float, input("Enter savings (Lakhs) separated by spaces: ").split()))
    y = list(map(int, input("Enter loan sanctions (0 or 1) separated by spaces: ").split()))
    return np.array(x1), np.array(x2), np.array(y)

# Get user input
x1, x2, y = get_user_input()

# Scale the data using Min-Max Scaling
x1_scaled = (x1 - np.min(x1)) / (np.max(x1) - np.min(x1))
x2_scaled = (x2 - np.min(x2)) / (np.max(x2) - np.min(x2))

# Initialize coefficients and learning rate
b0, b1, b2 = 0, 0, 0
alpha = 0.3
e = np.e  # Use numpy's constant for e

# Perform logistic regression
for _ in range(1000):  # Adjust the number of iterations as needed
    db0, db1, db2 = 0, 0, 0
    for i in range(len(x1_scaled)):
        # Calculate the prediction using the logistic function
        prediction = 1 / (1 + e ** (-(b0 + (b1 * x1_scaled[i]) + (b2 * x2_scaled[i]))))
        # Calculate the error
        error = y[i] - prediction
        # Update the gradients
        db0 += alpha * error * prediction * (1 - prediction)
        db1 += alpha * error * prediction * (1 - prediction) * x1_scaled[i]
        db2 += alpha * error * prediction * (1 - prediction) * x2_scaled[i]
    # Update the coefficients
    b0 += db0 / len(x1_scaled)
    b1 += db1 / len(x1_scaled)
    b2 += db2 / len(x1_scaled)

# Print the coefficients
print("B0 : ", b0, ", B1 : ", b1, ", B2 : ", b2)

# Create a meshgrid for plotting the decision boundary
x1_min, x1_max = min(x1_scaled), max(x1_scaled)
x2_min, x2_max = min(x2_scaled), max(x2_scaled)
xx1, xx2 = np.meshgrid(np.arange(x1_min - 0.1, x1_max + 0.1, 0.01),
                       np.arange(x2_min - 0.1, x2_max + 0.1, 0.01))

# Predict probabilities for the meshgrid points
Z = 1 / (1 + e ** (-(b0 + (b1 * xx1) + (b2 * xx2))))
Z = (Z > 0.5).astype(int)  # Convert probabilities to class labels

# Plot the decision boundary and sigmoid function
plt.figure(figsize=(10, 6))

# Plot the decision boundary
plt.contourf(xx1, xx2, Z, alpha=0.5, cmap='RdYlGn')
plt.scatter(x1_scaled, x2_scaled, c=y, cmap='RdYlGn', edgecolors='k', s=100)
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Savings (Scaled)')
plt.title('Logistic Regression Decision Boundary and Sigmoid Function')

# Plot the sigmoid function
x_sigmoid = np.linspace(-10, 10, 400)
y_sigmoid = 1 / (1 + np.exp(-x_sigmoid))
plt.plot(x_sigmoid, y_sigmoid, label='Sigmoid Function', color='blue', linewidth=2)

# Plot the decision threshold
plt.axhline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
plt.legend()

plt.tight_layout()
plt.show()
