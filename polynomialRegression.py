# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:40:18 2016

@author: rafaan
"""

# Fitting higher-order polynomial functions to visualize overfitting 

import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from copy import deepcopy

# Generating example data
n = 30
x = np.array([random.random() for i in range(n)])
x.sort(axis=0)
y = np.array([math.sin(4*i) for i in x])

# Adding noise to Y
e = np.array([random.gauss(0,1.0/5.0) for i in range(n)])
y = y + e

# Converting data to Pandas DataFrame as input to sk-learn models
data = pd.DataFrame(x, columns=['X1'])
data['Y'] = y

# Plotting function
def plot_data(data):    
    plt.plot(data['X1'], data['Y'],'k.')
    plt.xlabel('x')
    plt.ylabel('y')

def polynomial_features(data, deg):
    '''
    Returns a DataFrame with additional columns for each 
    degree of the polynomial specified by deg.
    
    Parameters
    ----------
    data : Pandas DataFrame
    deg : int representing the degree of the polynomial
    '''
    data_copy=data.copy()
    for i in range(1,deg):
        data_copy['X'+str(i+1)]=data_copy['X'+str(i)]*data_copy['X1']
    return data_copy
    
def polynomial_regression(data, deg):
    '''
    Returns a fitted model based on the speficied degree of the polynomial.
    
    Parameters
    ----------
    data : Pandas DataFrame
    deg : int representing the degree of the polynomial
    '''
    features = deepcopy(data)
    del features['Y']
    model = LinearRegression()
    model.fit(polynomial_features(features, deg), data['Y'])
    return model

def plot_poly_predictions(data, model):
    '''
    Returns a plot with the training data points and a line of 
    best fit according to the specified degrees of the polynomial.
    '''
    plot_data(data)

    # Get the degree of the polynomial
    deg = len(model.coef_)
    
    # Create 200 points in the x axis and compute the predicted value for each point
    x_pred = pd.DataFrame(np.linspace(0,1,200), columns=['X1'])
    y_pred = model.predict(polynomial_features(x_pred, deg))
    
    # plot predictions
    plt.plot(x_pred, y_pred, 'g-', label='degree ' + str(deg) + ' fit')
    plt.legend(loc='upper left')
    plt.axis([0,1,0,1])

# Generating and saving charts to visualize overfitting
def save_plots(data, degree_range):
    for i in range(degree_range + 1):
        plt.clf()
        model = polynomial_regression(data, deg=i)
        plot_poly_predictions(data, model)
        filename = 'polyreg' + str(i)
        plt.savefig(filename)
        
save_plots(data, 15)
