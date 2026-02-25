import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2]
y = dataset["Salary"]

# Training linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lr1 = LinearRegression()
lr1.fit(X,y)

# Training polynomial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(X_poly,y)
# print(X_poly)

# Visualising the Linear Regression result 
plt.scatter(X, y, color = 'red')
plt.plot(X, lr1.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression result 
plt.scatter(X, y, color = 'red')
plt.plot(X, lr2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()