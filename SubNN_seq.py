# a sequential bagNN that is only useful in experiment
# do fitting and prediction in one shot to avoid memory usage of models

# train more than one fwkNN model using subsets of data
# compare: this model  use subsample data to build tree and use entire data to train
#          boost_2     train a fkNN using subsample data

import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold


class SubNN_seq:
    ''' train multiple fwkNN with subsets of data, vote
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

    def fit(self, X, y, rSeed=1234):
        ''' X: array-like(numpy), training data, each row is a data point
            y: array-like, training data label/classification
        '''
        np.random.seed(rSeed)

        self.models = [KNeighborsClassifier( n_neighbors=1, algorithm=self.algorithm_NN) for i in range(self.n_model)]
        self.idx_set = [np.random.random_integers(low=0, high=len(y)-1, size=int(len(y)*self.subSampleRatio)) for i in range(self.n_model)]

        # fit a kNN to 'denoise' data
        knn_model = KNeighborsClassifier( n_neighbors=self.k_train, algorithm=self.algorithm_NN)
        knn_model.fit(X, y)
        y_pred = knn_model.predict(X)
        self.y_pred_train = y_pred
        self.X_train = X

    def predict(self, T):
        # prediction
        res_raw = [[] for i in range(T.shape[0])]
        for idx in self.idx_set:
            model = KNeighborsClassifier(n_neighbors=1, algorithm=self.algorithm_NN)
            model.fit(self.X_train[idx,], self.y_pred_train[idx])
            for i in range(T.shape[0]):
                res_raw[i].append(model.predict(T[[i],:])[0])
        # combine 
        res_final = [self.majority_element(res_i) for res_i in res_raw]
        return res_final



def test():
    np.random.seed(1)
    # general settings
    nSample = 20
    nAttribute = 2
    nClass = 3
    kCrossValidation = 2 # k-fold cross validation
    k_KNN = 5
    k_fkNN = 5

    # prepare data
    X = np.random.rand(nSample, nAttribute)
    y = np.random.random_integers(0, nClass-1, nSample)

    # a = bagNN_seq(subSampleRatio=0.3, weight_aggre='exp_err')
    a = SubNN_seq(k_train=1, n_model=2, subSampleRatio=1)
    a.fit(X, y)
    print( a.predict(X) )

    print('success!')



if __name__ == '__main__':
    test()