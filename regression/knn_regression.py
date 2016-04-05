# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:46:14 2016

@author: rafaan
"""
import numpy as np
from helper_functions import *
    
def distance(observation_q, observation_i):
    distance = np.sqrt(np.sum((observation_q - observation_i)**2))
    return distance

def 1_nearest_neighbor(features, query):
    distance = np.sqrt(np.sum((features - query)**2, axis=1))
    nearest_neighbor = min(distance)
    index = np.where(distance==nearest_neighbor)
    return features[index]

def k_nearest_neighbors_single_query(features, query, k=5):
    #compute distance
    #sort distances
    #get k closest features
    #return average weights
    return feature

def k_nearest_neighbors():
    #loop over rows in test set and compute k_nearest_neighbors_single_query for each

x = np.array([[1,1,1,1,1],[0,1,2,3,4],[2,3,4,5,6,],[3,9,12,15,18],[42,25,67,87,66],[1,4,8,14,22]]).T
y = np.array([1,3,7,13,21])

features, norms = normalize_features(x)