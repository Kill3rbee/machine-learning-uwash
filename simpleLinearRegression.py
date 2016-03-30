# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 22:28:18 2016

@author: rafaan
"""

# Closed-form solution to calculate slope & 
# intercept for simple linear regression
def simpleLinearRegression(input_feature, output):
    """
    Simple Linear Regression.
    
    Parameters
    ----------
    input_feature : array, list of input values
    output : array, list of target values
    """
    x = input_feature
    y = output
    
    sum_y = float(sum(y))
    sum_x = float(sum(x))
    sum_yx = float(sum([y[i] * x[i] for i in range(len(x))]))
    sum_x2 = float(sum([i**2 for i in x]))
    
    # Setting the gradient equal to zero
    w1 = (sum_yx - ((sum_y*sum_x)/len(x)))/(sum_x2 - ((sum_x*sum_x)/len(x)))
    w0 = (sum_y/len(x)) - ((w1*sum_x)/len(x))
    
    slope = w1
    intercept = w0
        
    return intercept, slope

def get_regression_predictions(input_feature, intercept, slope):
    """
    Get Regression Predictions.
    
    Parameters
    ----------
    input_feature : array, list of input values
    intercept : intercept value from simpleLinearRegression output
    slope : slope value from simpleLinearRegression output
    """
    
    x = input_feature
    
    predicted_values = [intercept + x[i]*slope for i in range(len(x))]
    
    return predicted_values
    
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    
    x = input_feature
    y = output
    
    y_hat = get_regression_predictions(x, intercept, slope)

    residuals = [(y[i] - y_hat[i])**2 for i in range(len(y))]
    RSS = sum(residuals)

    return(RSS)
    
# Testing output = 1 + 1*input_feature 
# because then we know both our slope 
# and intercept should be equal to 1,
# and the RSS will be 0.
train_x = [1, 2, 3, 4]
train_y = [2, 3, 4, 5]

slr = simpleLinearRegression(train_x, train_y)
intercept = slr[0]
slope = slr[1]

regression_predictions = get_regression_predictions(train_x, intercept, slope)
get_residual_sum_of_squares(train_x, train_y, intercept, slope)