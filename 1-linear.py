from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

# Input data for X and Y
x = list(map(int, input("Enter X Data : ").split(" ")))
y = list(map(int, input("Enter Y Data : ").split(" ")))

# Number of data points
n = len(x)

# Calculate the mean of X and Y
xmean = sum(x) / n
ymean = sum(y) / n

# Calculate the differences from the mean for X and Y
a = []
b = []
for i in range(n):
    a.append(x[i] - xmean)
    b.append(y[i] - ymean)

# Calculate the products of the differences and the squares of the differences
ab = [a[i] * b[i] for i in range(n)]
asquare = [a[i]**2 for i in range(n)]
bsquare = [b[i]**2 for i in range(n)]

# Calculate the correlation coefficient (r)
r = sum(ab) / sqrt(sum(asquare) * sum(bsquare))

# Calculate the standard deviations of X and Y
dely = sqrt(sum(bsquare)) / sqrt(n - 1)
delx = sqrt(sum(asquare)) / sqrt(n - 1)

# Calculate the slope (b1) and intercept (b0) of the regression line
b1 = r * dely / delx
b0 = ymean - b1 * xmean

# Print the coefficients and the regression equation
print("B0 : ", b0, ", B1 : ", b1)
print("Equation : y=", b0, "+", b1, "x")

# Plotting the data points and regression line
sns.scatterplot(x=x, y=y)  # Scatter plot for data points

# Generate x values for prediction (smooth line)
xpred = [i for i in range(min(x), max(x) + 1)]  
ypred = [b0 + (b1 * i) for i in xpred]  # Predict y values using the equation

# Plot the regression line
sns.lineplot(x=xpred, y=ypred)  # Line plot for regression line
plt.show()

# Wait for user input to exit
input("Press Enter to exit...")