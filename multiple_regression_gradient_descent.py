# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:03:21 2016

@author: rafaan
"""

import numpy as np
from math import sqrt

# Defining the helper functions and multiple regression algorithm
def predict_output(feature_matrix, weights):
    
    """
    Predict Output.
    
    Takes in a feature matrix and weights and
    returns predicted values using dot product.
    
    Parameters
    ----------
    feature_matrix : 2d array
    weights: 1 d vector
    """
    
    predictions = np.dot(feature_matrix, weights)
    return(predictions)

def feature_derivative(errors, feature):
    
    """
    Feature derivative.
    
    Parameters
    ----------
    errors : 1d array
    features: 1d vector
    """
    
    derivative = 2*np.dot(errors, feature)
    return(derivative)
    
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    
    """
    Runs gradient descent on multiple regression.
    
    Parameters
    ----------
    feature_matrix : 2d array
    output: 1d array
    initial_weights: 1 d vector
    step_size = int/float value representing eta or magnitude of update to coefficients/weights at a given iteration
    tolerance = int/float value representing the convergence criteria
    """
    
    converged = False 
    weights = initial_weights # make sure it's a numpy array
    while not converged:
        # compute the predictions based on feature_matrix and weights using predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output
        gradient_sum_squares = 0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors, feature_matrix[:, i])
            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares = derivative**2
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - (step_size*derivative)
        # compute the square-root of the gradient sum of squares to get the gradient matnigude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)
    
# Generating example data and setting initial gradient descent parameters
constant = np.ones(5)
x1 = np.array([0,1,2,3,4])
x2 = np.array([1,2,3,4,5])
y = np.array([1,3,7,13,21])
data = np.array([constant, x1, x2]).T
initial_weights = np.zeros(3)
step_size = .005
tolerance = .01

# Fitting the model using the above data and parameters 
regression_gradient_descent(data, y, initial_weights, step_size, tolerance)

