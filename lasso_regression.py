# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 13:54:50 2016

@author: rafaan
"""

import numpy as np
import math

def normalize_features(feature_matrix):
    '''
    Normalize Features.
    
    Returns 2d array with normalized features

    Parameters
    ----------
    feature_matrix : 2d array
    '''
    norms = np.linalg.norm(feature_matrix, axis=0)
    features = feature_matrix/norms
    return features, norms

def predict_output(feature_matrix, weights):
    
    """
    Predict Output.
    
    Takes in a feature matrix and weights and
    returns predicted values using dot product.
    
    Parameters
    ----------
    feature_matrix : 2d array
    weights: 1d array
    """
    
    predictions = np.dot(feature_matrix, weights)
    return(predictions)
    
def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    '''
    Lasso Coordinate Descent Step.

    Returns new weight for feature[i] by calculating the residual of the model
    without feature[i] and setting it equal to ro_i.  If ro_i is between -l1_penalty/2
    and +l1_penalty/2 then the new weight will be set to 0, otherwise it will be set 
    equal to ro_i plus/minus l1_penalty/2 (lambda).

    Parameters
    ----------
    i : int representing the ith feature column of a matrix
    feature_matrix : normalized 2d array
    output : array of target values
    weights: 1d array
    l1_penalty : float/int value representing the l1 norm penalty
    '''    
    # compute prediction
    prediction = predict_output(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    # ro_i = sum of feature[i] times the residual without feature[i], 
    #   where residual = (actual - predictions(w/o feature i)
    # ro_i is the partial derivative of the RSS cost function with respect to w[i],
    #   where we're fixing all other w's and taking the partial derivative wrt w[i].
    #   In other words, it's the correlation between feature[i] and residuals of the 
    #   model without feature[i], which measures the strength of feature[i] to the model. 
    
    ro_i = sum(feature_matrix[:,i]*(output - prediction + weights[i]*feature_matrix[:,i]))

    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i 
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0.
    
    return new_weight_i
    
def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    '''
    Lasso Cyclical Coodinate Descent.
    
    Returns the new weights of the lasso regression.
    
    Parameters
    ----------
    feature_matrix : normalized 2d array
    output : array of target values
    initial_weights: 1d array
    l1_penalty : float/int value representing the l1 norm penalty
    tolerance : float/int value representing the convergence criteria    
    '''    
    weights = initial_weights
    converged = False
    
    while not converged:
        coordinate_changes = []
        for i in range(len(weights)):
            old_weights_i = weights[i]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            coordinate_changes.append(old_weights_i - weights[i])
        if max(coordinate_changes) < tolerance:
            converged = True
    return weights

# Generating example data
x = np.array([[1,1,1,1,1],[0,1,2,3,4],[2,3,4,5,6,],[3,9,12,15,18],[42,25,67,87,66],[1,4,8,14,22]]).T
y = np.array([1,3,7,13,21])
initial_weights = np.zeros(6)
l1_penalty = .01
tolerance = 1.0

# Normalizing features and fitting lasso regression
features, norms = normalize_features(x)
lasso_cyclical_coordinate_descent(features, y, initial_weights, l1_penalty, tolerance)