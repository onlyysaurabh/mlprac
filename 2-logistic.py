import matplotlib.pyplot as plt
import numpy as np

# Input data (replace with your actual data)
x1 = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])  # Annual income (Lakhs)
x2 = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])  # Savings (Lakhs)
y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1])  # Loan sanction (0 or 1)

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
        prediction = 1 / (1 + e ** (-(b0 + (b1 * x1_scaled[i]) + (b2 * x2_scaled[i]))))
        error = y[i] - prediction
        db0 += alpha * error * prediction * (1 - prediction)
        db1 += alpha * error * prediction * (1 - prediction) * x1_scaled[i]
        db2 += alpha * error * prediction * (1 - prediction) * x2_scaled[i]
    b0 += db0 / len(x1_scaled)
    b1 += db1 / len(x1_scaled)
    b2 += db2 / len(x1_scaled)

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

plt.axhline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')  # Decision threshold
plt.legend()

plt.tight_layout()
plt.show()
