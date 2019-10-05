#!/usr/bin/env python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import common.feature_selection as feat_sel
import common.test_env as test_env

# Imports for visualization
import matplotlib.pyplot as plt

# Imports for Polynomial Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Imports for Support Vector Regression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np

# Imports for Decision Tree
from sklearn.tree import DecisionTreeRegressor

# Import for Random Forest
from sklearn.ensemble import RandomForestRegressor


def print_metrics(y_true, y_pred, label):
    from sklearn.metrics import r2_score
    # Feel free to extend it with additional metrics from sklearn.metrics
    print('%s R squared: %.2f' % (label, r2_score(y_true, y_pred)))


def linear_regression(X, y, print_text='Linear regression all in'):
    # Split train test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Linear regression all in
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print_metrics(y_test, reg.predict(X_test), print_text)
    return reg


def linear_regression_selection(X, y):
    X_sel = feat_sel.backward_elimination(X, y)
    return linear_regression(X_sel, y, print_text='Linear regression with feature selection')


def polynomial_regression(X, y , print_text='Polynomial Regression all in'):
    poly_reg = PolynomialFeatures(degree=1)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)

    print_metrics(y, pol_reg.predict(X_poly), print_text)

    return pol_reg


def support_vector_regression(X, y, print_text='SVR all in'):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = np.ravel(sc.fit_transform(np.expand_dims(y, axis=1)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=0)

    sv = SVR(gamma='scale', kernel='rbf')
    sv.fit(X_test, y_test)

    print_metrics(np.squeeze(y_test), np.squeeze(sv.predict(X_test)), print_text)

    return sv


def decision_tree_regression(X,  y, print_text='Decision Tree all in'):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=0)

    dt = DecisionTreeRegressor(max_depth=None,
                                min_samples_split=85,
                                min_samples_leaf=70)
    dt.fit(X_train, y_train)
    print_metrics(y_train, dt.fit(X_train, y_train).predict(X_train), print_text)
    return dt


def random_forest_regression(X, y, print_text='Random Forest all in'):
    rf = RandomForestRegressor(n_estimators=1)
    rf.fit(X, y)
    print_metrics(y, rf.fit(X, y).predict(X), print_text)
    return rf

def plot(X,y, type):
    if type == 'plot':
        plt.plot(X, y, color='red')
        plt.show()
    if type == 'scatter':
        plt.scatter(X, y, color='red')
        plt.show()
    return

if __name__ == '__main__':
    REQUIRED = ['numpy', 'statsmodels', 'sklearn']
    test_env.versions(REQUIRED)

    X, y = load_boston(return_X_y=True)

    linear_regression(X, y)
    linear_regression_selection(X, y)
    polynomial_regression(X, y)
    support_vector_regression(X, y)
    decision_tree_regression(X, y)
    random_forest_regression(X, y)

'''
RESULT
# Python and modules versions
Python: 3.7.4 (default, Aug 13 2019, 15:17:50) 
[Clang 4.0.1 (tags/RELEASE_401/final)]
numpy: 1.17.2
statsmodels: 0.10.1
sklearn: 0.21.3


Linear regression all in R squared: 0.64
Linear regression with feature selection R squared: 0.64
Polynomial Regression all in R squared: 0.91
SVR all in R squared: 0.21
Decision Tree all in R squared: 1.00
Random Forest all in R squared: 0.98
'''

'''
Expected RESULT:
# Python and modules versions
Python: 3.7.3 (default, Mar 27 2019, 16:54:48) 
[Clang 4.0.1 (tags/RELEASE_401/final)]
numpy: 1.16.2
statsmodels: 0.9.0
sklearn: 0.20.3

Linear regression all in R squared: 0.64
Linear regression with feature selection R squared: 0.64
Polynomial regression R squared: 0.55
SVR R squared: 0.71
Decision Tree regression R squared: 0.61
Random forest regression R squared: 0.75
Done
'''

