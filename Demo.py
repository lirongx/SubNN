#!#!/usr/bin/env python3

# Demo of SubNN package
# Lirong Xue
# Jun 11, 2017

from SubNN import SubNN
import numpy as np

# general settings
nSample = 200
nAttribute = 2
nClass = 3

# create Gaussian Mixture data
cov = np.identity(nClass-1)
means = np.zeros((nClass, nClass-1))
for i in range(nClass-1):
    means[i, i] = 2
prob_acc = np.linspace(0, 1, nClass)
X = np.zeros((nSample, nClass-1))
y = np.zeros((nSample))
for i in range(nSample):
    tmp = np.random.random()
    for label in range(nClass):
        if tmp >= prob_acc[label]:
            y[i] = label
            X[i,:] = np.random.multivariate_normal(means[label,], cov)


a = SubNN(k_train=10, subSampleRatio=1)
a.fit(X, y)
# print( a.predict(X) )
print( a.cross_validate_k(X, y, 6) )
print( a.search_k(X,y) )
print( a.error_rate(y, a.predict(X)))

print('success!')



''' Problems:
    1. if run on 1 computer: huge memory consumption
    2. if run on multiple computer: how to comunicate?
    3. moreover: how to find the best k? on 1 com or >1? In para (faster) or in sequential (much slower, used in NIPS)
'''