import numpy as np
from math import sqrt

# Generating example data and setting initial gradient descent parameters
x = np.array([0,1,2,3,4])
y = np.array([1,3,7,13,21])
data = np.array([x, y])
initial_weights = np.array([0])
step_size = .05
tolerance = .01

def simpleLinearRegressionGradientDescent(data, step_size, tolerance):
    '''
    Simple Linear Regression Using Gradient Descent
    
    
    '''
    converged = False
    intercept = 0
    slope = 0
    while not converged:
       
        prediction = (slope * data[0]) + np.repeat(intercept, 5)
        errors = prediction - data[1]

        # update intercept
        derivative_intercept = sum(errors)
        intercept = intercept - (step_size*derivative_intercept)
        # update slope
        derivative_slope = sum(errors*data[0])
        slope = slope - (step_size*derivative_slope)
        
        gradient_sum_squares = derivative_intercept**2 + derivative_slope**2
        magnitude = sqrt(gradient_sum_squares)
        
        if magnitude < tolerance:
            converged = True
    return(intercept, slope)

# Fitting the model using the above data
simpleLinearRegressionGradientDescent(data, step_size, tolerance)
