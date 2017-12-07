#!#!/usr/bin/env python3

# SubNN class
# Lirong Xue
# Jun 11, 2017


import numpy as np
import multiprocessing as mp
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

    # calculate error rate of a given k through cross validation
    def cross_validate_k(self, X, y, k, n_folds=2, n_repeat=1):
        scores= []
        for repeat in range(n_repeat):
            skf = StratifiedKFold(y, n_folds = 2, shuffle=True)

            for train_index, test_index in skf:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if k > len( y_train ): k = len( y_train )
                self.k_train = k
                self.fit(X_train, y_train)
                scores.append( self.error_rate(y_test, self.predict(X_test)) )

        return np.mean(scores)

    # search for a optimal k and fit
    def search_k(self, X, y, n_folds=2, n_repeat=1):
        # search best k in 1,2,4,8,16,...
        k_set = []
        k_err = []
        k = 1
        while k < X.shape[0]:
            k_set.append( k )
            k_err.append( self.cross_validate_k(X, y, k, n_folds=n_folds, n_repeat=n_repeat) )
            k = k * 2
        k_opt_rough = k_set[ np.argmin(k_err) ]

        # search for optimal k in [k_opt_rough/2, k_opt_rough*2]
        for k in range( max(1, int(k_opt_rough/2)), min(k_opt_rough*2, X.shape[0]) ):
            if k not in k_set:
                k_set.append( k )
                k_err.append( self.cross_validate_k(X, y, k, n_folds=n_folds, n_repeat=n_repeat) )
        
        k_opt = k_set[ np.argmin(k_err) ]
        self.k_train = k_opt
        return k_opt

    # fit model, settings are given at initialization
    def fit(self, X, y, rSeed=1234):
        ''' X: array-like(numpy), training data, each row is a data point
            y: array-like, training data label/classification
        '''
        np.random.seed(rSeed)

        self.models = [KNeighborsClassifier( n_neighbors=1, algorithm=self.algorithm_NN) for i in range(self.n_model)]

        # fit a kNN to 'denoise' data
        knn_model = KNeighborsClassifier( n_neighbors=self.k_train, algorithm=self.algorithm_NN)
        knn_model.fit(X, y)
        y_pred = knn_model.predict(X)
        
        for i in range(self.n_model):
            idx = np.random.random_integers(low=0, high=len(y)-1, size=int(len(y)*self.subSampleRatio))
            self.models[i].fit(X[idx,], y_pred[idx])

    def predict(self, T, parallel=True):
        if not parallel:
            res_models = [mdl.predict(T) for mdl in self.models]
        else: # run the code parallelly
            # def do(mdl):
            #     return mdl.predict(T)
            with mp.Pool() as pool:
                # print(pool.map(f, range(10)))
                # res_models = pool.map(do, self.models)
                res_models = pool.map(do, zip(self.models, [T]*self.n_model))
                print(self.n_model)

        res_final = [None for i in range(T.shape[0])]
        for i in range(T.shape[0]):
            res_final[i] = self.majority_element([res[i] for res in res_models])
        return res_final

def do(things):
    return things[0].predict(things[1])




def test():
    np.random.seed(1)
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
    a = SubNN(k_train=1, n_model=2, subSampleRatio=1)
    a.fit(X, y)
    print( a.predict(X) )
    # print( a.cross_validate_k(X, y, 6) )
    # print( a.search_k(X,y) )
    print( a.error_rate(y, a.predict(X)))

    print('success!')

if __name__ == '__main__':
    test()