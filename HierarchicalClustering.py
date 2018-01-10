#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:54:42 2018

@author: virajdeshwal
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

file = pd.read_csv('Mall_Customers.csv')
X= file.iloc[:,[3,4]].values


''' We will use the dendrograms to visualize the optimal no. of clusters'''

#Use dendrograms to find the optimal number of clusters.

import scipy.cluster.hierarchy as sch
dendrograms =sch.dendrogram(sch.linkage(X, method ='ward'))
plt.title('Dendrograms')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

print('We got the optimal number of clusters from the Dendrograms\nNow lets fit HC to data\n.')
inp = input('Press any key to continue....\n')
'''We got the number of clusters. And now we will define the optimal number of clusters by 
counting the number of lines which are parellely to the largest vertical dendride.
Here we have 5 lines along with the largest vertical dendride.
So the optimal number of clusters will be 5.'''

#Fitting HC to the dataset

from sklearn.cluster import AgglomerativeClustering
model= AgglomerativeClustering(n_clusters = 5, affinity='euclidean', linkage= 'ward')
y_hc = model.fit_predict(X)


#Now lets visualize the clusters

plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s=100, c='red', label = 'Careful pals')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s=100, c='blue', label = 'average')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s=100, c='green', label = 'Targets')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s=100, c='magenta', label = 'Freak')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s=100, c='cyan', label = 'Sensible')

#We do not need centroids in Hierarchical clustering.

plt.title('clusters of client')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print('\nDone ;)')