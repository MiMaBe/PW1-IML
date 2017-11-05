import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
import sys
import pylab as plt
from kMeans import kMeans
import pandas as pd

def show(X, C, centroids, keep=False):
    plt.cla()
    plt.plot(X[C == 0, 0], X[C == 0, 1], '*b',
         X[C == 1, 0], X[C == 1, 1], '*r',
         X[C == 2, 0], X[C == 2, 1], '*g')
    plt.plot(centroids[:, 0], centroids[:, 1], '*m', markersize=20)
    plt.draw()
    if keep :
        plt.ioff()
        plt.show()

adultData, adultMeta = arff.loadarff('/Users/mimabe/Documents/workspace/PW1-IML/data/adult.arff')
irisData, irisMeta = arff.loadarff('/Users/mimabe/Documents/workspace/PW1-IML/data/iris.arff')
wineData, irisMeta = arff.loadarff('/Users/mimabe/Documents/workspace/PW1-IML/data/wine.arff')
balData, balMeta = arff.loadarff('/Users/mimabe/Documents/workspace/PW1-IML/data/bal.arff')

balData = pd.DataFrame(balData)

msk = np.random.rand(len(balData)) <= 0.7
balDataTrain = balData[msk]
balDataTest = balData[~msk]

le = LabelEncoder()

for col in balDataTest.columns.values:
    # Encoding only categorical variables
    if balDataTest[col].dtypes == 'object':
        # Using whole data to form an exhaustive list of levels
        data = balDataTrain[col].append(balDataTest[col])
        le.fit(data.values)
        balDataTrain[col] = le.transform(balDataTrain[col])
        balDataTest[col] = le.transform(balDataTest[col])
        
plt.ion()

centroids, C = kMeans(balDataTrain.values.astype(int), K=3, plot_progress=show)
print C
show(balData.values, C, centroids, True)
