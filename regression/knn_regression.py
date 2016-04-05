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

def distances(features, query):
    distances = np.sqrt(np.sum((features - query)**2, axis=1))
    return distances

def one_nearest_neighbor(features, query):
    distance_to_nearest_neighbor = float('inf')
    nearest_neighbor = []
    for i in range(len(features)):
        test_distance = distance(features[i], query)
        if test_distance < distance_to_nearest_neighbor:
            nearest_neighbor = features[i]
            distance_to_nearest_neighbor = test_distance
    return nearest_neighbor

def fetch_k_nearest_neighbors(features, query, k=5):
    distances_array = distances(features, query)
    sorted_order = np.argsort(distances_array)
    index = np.where(sorted_order <= k)

    #sort distances
    #get k closest features
    return indices

def k_nearest_neighbors():
    #loop over rows in test set and compute k_nearest_neighbors_single_query for each

x = np.array([[1,1,1,1,1],[2,3,4,5,6,],[3,9,12,15,18],[0,1,2,3,4],[42,25,67,87,66],[1,4,8,14,22]]).T
y = np.array([1,3,7,13,21])

features, norms = normalize_features(x)
