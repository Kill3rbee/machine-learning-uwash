# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:46:14 2016

@author: rafaan
"""

from helper_functions import *
    
def distance_to_nearest_neighbor(observation_q, observation_i, distance_metric = 'euclidean'):
    if distance_metric == 'euclidean':
        distance = np.linalg.norm(observation_q - observation_i)
    return distance

x = np.array([[1,1,1,1,1],[0,1,2,3,4],[2,3,4,5,6,],[3,9,12,15,18],[42,25,67,87,66],[1,4,8,14,22]]).T
y = np.array([1,3,7,13,21])

normalize_features(x)
dist = np.linalg.norm(x[1]-x[2])