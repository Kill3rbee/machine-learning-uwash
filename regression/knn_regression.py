# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:46:14 2016

@author: rafaan
"""
import numpy as np
from helper_functions import *
    
def distance_to_nearest_neighbor(observation_q, observation_i, distance_metric = 'euclidean'):
    if distance_metric == 'euclidean':
        distance = np.sqrt(np.sum((observation_q - observation_i)**2))
    return distance

x = np.array([[1,1,1,1,1],[0,1,2,3,4],[2,3,4,5,6,],[3,9,12,15,18],[42,25,67,87,66],[1,4,8,14,22]]).T
y = np.array([1,3,7,13,21])

features, norms = normalize_features(x)

distance_to_nearest_neighbor(np.array([1,1,1,1,1]), np.array([1,1,1,1,1]))

results = features[0:3] - features[4]
print results[0] - (features[0]-features[4])
# should print all 0's if results[0] == (features_train[0]-features_test[0])
print results[1] - (features[1]-features[4])
# should print all 0's if results[1] == (features_train[1]-features_test[0])
print results[2] - (features[2]-features[4])