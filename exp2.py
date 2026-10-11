# OG OG OG OG
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

print("ASWIN PV-15")

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"

df = pd.read_csv(url)
df = df[["displacement", "mpg"]].dropna()
print(df.head())

X = df[["displacement"]]
y = df["mpg"]

# Linear
linear = LinearRegression()
linear.fit(X, y)
y_linear = linear.predict(X)

# Polynomial degree 2
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)
model2 = LinearRegression()
model2.fit(X_poly2, y)
y_poly2 = model2.predict(X_poly2)

# Polynomial degree 3
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X)
model3 = LinearRegression()
model3.fit(X_poly3, y)
y_poly3 = model3.predict(X_poly3)

print("\nLinear Regression")
print("MSE :", mean_squared_error(y, y_linear))
print("R2 Score :", r2_score(y, y_linear))

print("\nPolynomial Regression (Degree 2)")
print("MSE :", mean_squared_error(y, y_poly2))
print("R2 Score :", r2_score(y, y_poly2))

print("\nPolynomial Regression (Degree 3)")
print("MSE :", mean_squared_error(y, y_poly3))
print("R2 Score :", r2_score(y, y_poly3))

# Smooth curve for plotting
x_plot = pd.DataFrame(
    np.linspace(X["displacement"].min(), X["displacement"].max(), 300),
    columns=["displacement"]
)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="black", label="Data")
plt.plot(x_plot, linear.predict(x_plot), color="red", label="Linear Regression")
plt.plot(x_plot, model2.predict(poly2.transform(x_plot)), label="Polynomial Degree 2")
plt.plot(x_plot, model3.predict(poly3.transform(x_plot)), label="Polynomial Degree 3")

plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Polynomial Regression vs Linear Regression")
plt.legend()
plt.grid(True)
plt.show()