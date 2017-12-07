#!#!/usr/bin/env python3

# SubNN class
# Lirong Xue
# Jun 11, 2017


import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold


class SubNN:
    ''' subNN model, use bagged denoised 1NN for prediction
    '''

    def __init__(self, k_train=10, n_model=10, subSampleRatio=0.1, algorithm_NN='kd_tree'):
        self.k_train = k_train; # 'k' of k-NN used in denoising
        self.n_model = n_model
        self.subSampleRatio = subSampleRatio # alpha (ratio of subsampling)
        self.algorithm_NN = algorithm_NN

    # find majority element is a sequence
    def majority_element(self, seq):
        c = Counter(seq)
        value, count = c.most_common(1)[0]
        return value

    def error_rate(self, truth, prediction):
        wrong = 0
        for x, y in zip(truth, prediction):
            if x != y:
                wrong = wrong + 1
        return wrong / len(truth)

    # fit model, settings are given at initialization
    def fit(self, X, y, rSeed=1234):
        ''' X: array-like(numpy), training data, each row is a data point
            y: array-like, training data label/classification
        '''
        np.random.seed(rSeed)

        self.models = [KNeighborsClassifier( n_neighbors=1, algorithm=self.algorithm_NN) for i in range(self.n_model)]
        self.weights_aggre = [1 for x in range(self.n_model)]

        # fit a kNN to 'denoise' data, using k = k_train
        knn_model = KNeighborsClassifier( n_neighbors=self.k_train, algorithm=self.algorithm_NN)
        knn_model.fit(X, y)
        y_pred = knn_model.predict(X)
        
        for i in range(self.n_model):
            idx = np.random.random_integers(low=0, high=len(y)-1, size=int(len(y)*self.subSampleRatio))
            self.models[i].fit(X[idx,], y_pred[idx])

    # a function to predict sun1NN results
    def predict_raw(self, T):
        res_models = [mdl.predict(T) for mdl in self.models]
        return res_models

    # a function to combine a list of sub1NN results
    def predict_combine(self, result_list):
        n_data = len(result_list[0])
        res_final = [None for i in range(n_data)]
        for i in range(n_data):
            res_final[i] = self.majority_element([res[i] for res in result_list])
        return res_final

    # a function to combine a list of sub1NN results
    def predict(self, T):
        res_models = [mdl.predict(T) for mdl in self.models]
        n_data = len(res_models[0])
        res_final = [None for i in range(n_data)]
        for i in range(n_data):
            res_final[i] = self.majority_element([res[i] for res in res_models])
        return res_final



def test():
    # general settings
    nSample = 200
    nAttribute = 2
    nClass = 3
    kCrossValidation = 2 # k-fold cross validation
    k_KNN = 5
    k_fkNN = 5

    # prepare data
    X = np.random.rand(nSample, nAttribute)
    y = np.random.random_integers(0, nClass-1, nSample)
    a = SubNN(k_train=10, subSampleRatio=1)
    a.fit(X, y)
    print( a.predict(X) )
    print( a.error_rate(y, a.predict(X)))

    X1 = np.random.rand(nSample, nAttribute)
    y1 = np.random.random_integers(0, nClass-1, nSample)
    a1 = SubNN(k_train=10, subSampleRatio=1)
    a1.fit(X1, y1)
    a_res = a.predict_raw(X)
    a1_res = a1.predict_raw(X1)
    res_final = a.predict_combine(a_res + a1_res)
    print( res_final )
   

    print('success!')



if __name__ == '__main__':
    test()