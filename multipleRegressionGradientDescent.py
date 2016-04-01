# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:03:21 2016

@author: rafaan
"""

import numpy as np

def get_numpy_data(dataframe, feature_names, output_name):
    
    """
    Get Numpy Data.
    
    Converts data to numpy arrays and
    adds a column for the constant term.
    
    Parameters
    ----------
    dataframe : pandas dataframe
    feature_names : array, list of strings
    output_name: string
    """
    
    dataframe['constant'] = 1
    feature_names = ['constant'] + feature_names
    feature_dataframe = dataframe[feature_names]
    feature_matrix = np.array(feature_dataframe)
    output_array = dataframe[output_name]
    output_array = np.array(output_array)
    return(feature_matrix, output_array)
    
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
    step_size = eta, small sizes
    tolerance = stopping criteria
    """
    
    converged = False 
    weights = np.array(initial_weights) # make sure it's a numpy array
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
            gradient_sum_squares += derivative**2
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - (step_size*derivative)
        # compute the square-root of the gradient sum of squares to get the gradient matnigude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)
    
# Test
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)