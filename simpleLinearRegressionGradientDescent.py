import numpy as np
from math import sqrt
x = np.array([0,1,2,3,4])
y = np.array([1,3,7,13,21])
data = np.array([x, y])
initial_weights = np.array([0])
step_size = .05
tolerance = .01

def simpleLinearRegressionGradientDescent(data, step_size, tolerance):
    converged = False
    intercept = 0
    slope = 0
    while not converged:
       
        prediction = (slope * data[0]) + np.repeat(intercept, 5)
        errors = prediction - data[1]

        # update intercept
        derivative_intercept = sum(errors)
        intercept = intercept - (step_size*derivative_intercept)
        #print intercept
        derivative_slope = sum(errors*data[0])
        slope = slope - (step_size*derivative_slope)
        #print slope
        gradient_sum_squares = derivative_intercept**2 + derivative_slope**2
        magnitude = sqrt(gradient_sum_squares)
        #print magnitude

        if magnitude < tolerance:
            converged = True
    return(intercept, slope)
   
simpleLinearRegressionGradientDescent(data, step_size, tolerance)
