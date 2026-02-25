import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Importing data 
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.drop('Salary',axis = 1)
y = dataset['Salary']

# Train_Test_Split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
# Fit tells here are x and y build or figure out a relationship between them
Regressor.fit(X_train,y_train)
y_pred = Regressor.predict(X_test)

# Visualising training set result 
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train, Regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising test set result 
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train, Regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print(Regressor.predict([[10.5]]))
print(Regressor.coef_)
print(Regressor.intercept_)

# Salary = coefficient * X + intercept