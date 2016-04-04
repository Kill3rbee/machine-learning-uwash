# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 21:37:32 2016

@author: rafaan
"""

import numpy as np

# Helper functions

def normalize_features(feature_matrix):
    '''
    Normalize Features.  Note: Should normalize 
    train, validation, and test sets, otherwise
    the regression weights must be rescaled by 
    dividing them by the norms output from this 
    function before applying them to the unnormalized
    validation or test sets.
    
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