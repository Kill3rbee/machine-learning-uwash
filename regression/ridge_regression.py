# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:03:21 2016

@author: rafaan
"""

import numpy as np
from helper_functions import *

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    
    """
    Feature Derivative for Ridge Regression.
    
    Parameters
    ----------
    errors : 1d array
    feature : 1d vector
    weight : int/float representing initial weight or coefficient estimate from regression
    l2_penalty : int/float representing l2 norm penalty 
    feature_is_constant : boolean indicating whether feature is a constant
    """
    if feature_is_constant:
        derivative = 2*np.dot(errors, feature)
    else:
        derivative = 2*np.dot(errors, feature) + 2*l2_penalty*weight
    return(derivative)
    
def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    
    """
    Ridge Regression with Gradient Descent.
    
    Parameters
    ----------
    feature_matrix : 2d array
    output: 1d array
    initial_weights: 1d array
    step_size = int/float value representing eta or magnitude of update to coefficients/weights at a given iteration
    tolerance = int/float value representing the convergence criteria
    """
    weights = initial_weights
    iterations = 0
    while iterations <= max_iterations:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output
        
        for i in xrange(len(weights)): # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            #(Remember: when i=0, you are computing the derivative of the constant!)
            if i == 0:
                derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, True)
            else:
                derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, False)
            # subtract the step size times the derivative from the current weight    
            weights[i] = weights[i] - (step_size*derivative)
        iterations += 1
    return(weights)
    
# Generating example data and setting initial gradient descent parameters
constant = np.ones(5)
x1 = np.array([0,1,2,3,4])
x2 = np.array([1,2,3,4,5])
y = np.array([1,3,7,13,21])
data = np.array([constant, x1, x2]).T
initial_weights = np.zeros(3)
step_size = .005

# Fitting the model using the above data and parameters 
ridge_regression_gradient_descent(data, y, initial_weights, step_size, 0)
