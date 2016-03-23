# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 22:28:18 2016

@author: rafaan
"""

# Closed-form solution to calculate slope & 
# intercept for simple linear regression
def simpleLinearRegression(x, y):
    """
    Simple Linear Regression.
    
    Parameters
    ----------
    x : array, list of input values
    y : array, list of target values
    """
    sum_y = float(sum(y))
    sum_x = float(sum(x))
    sum_yx = float(sum([y[i] * x[i] for i in range(len(x))]))
    sum_x2 = float(sum([i**2 for i in x]))
    
    # Setting the gradient equal to zero
    w1 = (sum_yx - ((sum_y*sum_x)/len(x)))/(sum_x2 - ((sum_x*sum_x)/len(x)))
    w0 = (sum_y/len(x)) - ((w1*sum_x)/len(x))
    
    slope = w1
    intercept = w0
        
    return (intercept, slope)
    
# Testing output = 1 + 1*input_feature 
# because then we know both our slope 
# and intercept should be equal to 1.
simpleLinearRegression([1,2,3,4], [2,3,4,5])