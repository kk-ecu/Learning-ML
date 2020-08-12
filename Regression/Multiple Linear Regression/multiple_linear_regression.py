# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
# matrix of feature independent variable(country, age & salary)
X = dataset.iloc[:, :-1].values
# matrix of dependent variable
y = dataset.iloc[:, -1].values

# Encoding categorical data(State)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
columntransform = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = columntransform.fit_transform(X)

# Avoiding Dummy Variable Trap
X = X[:, 1:]

# Splitting the data set into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set
y_pred = regressor.predict(X_test)

# Building the optimal model using backward Elimination
import statsmodels.api as sm

# Assuming significance levels as 0.05
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_optimal = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# remove independent variable at col 2 New York(State)
X_optimal = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# remove independent variable at col 1 California(State)
X_optimal = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# remove independent variable at col 4 (Administration)
X_optimal = np.array(X[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# remove independent variable at col 5 (Marketing Spend)
X_optimal = np.array(X[:, [0, 3]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()