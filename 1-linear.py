from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

x = list(map(int, input("Enter X Data : ").split(" ")))
y = list(map(int, input("Enter Y Data : ").split(" ")))
n = len(x)
xmean = sum(x) / n
ymean = sum(y) / n

a = []
b = []
for i in range(n):
    a.append(x[i] - xmean)
    b.append(y[i] - ymean)

ab = [a[i] * b[i] for i in range(n)]
asquare = [a[i]**2 for i in range(n)]
bsquare = [b[i]**2 for i in range(n)]

r = sum(ab) / sqrt(sum(asquare) * sum(bsquare))
dely = sqrt(sum(bsquare)) / sqrt(n - 1)
delx = sqrt(sum(asquare)) / sqrt(n - 1)
b1 = r * dely / delx
b0 = ymean - b1 * xmean

print("B0 : ", b0, ", B1 : ", b1)
print("Equation : y=", b0, "+", b1, "x")

# Plotting the data points and regression line
sns.scatterplot(x=x, y=y)  # Scatter plot for data points

# Generate x values for prediction (smooth line)
xpred = [i for i in range(min(x), max(x) + 1)]  
ypred = [b0 + (b1 * i) for i in xpred]  # Predict y values using the equation

sns.lineplot(x=xpred, y=ypred)  # Line plot for regression line
plt.show()
input("Press Enter to exit...")