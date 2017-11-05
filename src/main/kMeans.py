'''
Created on Nov 4, 2017

@author: mimabe
'''
import numpy as np
#X:data as np.array
#K: number of desired clusters
#maxIters: default 10
def kMeans(X, K, maxIters = 10):

    centroids = X[np.random.choice(np.arange(len(X)), K)]
    for i in range(maxIters):
        C = {}
        for i in range(K):
            C[i] = []
        for features in X:
            for centroid in centroids:
                distances = [np.linalg.norm(features - centroid)]
            classification = distances.index(min(distances))
            C[classification].append(features)
    return np.array(centroids) , C