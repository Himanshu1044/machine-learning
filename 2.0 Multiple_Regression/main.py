import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing data 
dataset = pd.read_csv('50_Startups.csv')
X = dataset.drop('Profit', axis = 1)
y = dataset['Profit']

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
CT = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3] )], remainder='passthrough')
X = np.array(CT.fit_transform(X))

# Train-Test-Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Training multiple lienar regression model on training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predecting test result
y_pred = lr.predict(X_test)
np.set_printoptions(precision=2)
print(np.column_stack((y_pred, y_test)))

