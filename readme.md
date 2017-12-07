# SubNN
*Achieving the time of 1-NN, but the accuracy of k-NN*

### Description
  This is the code we used for our experiment in [our paper (submitted to NIPS 2017)](www.arxiv.org/sdfsfsfafsd). We propose a simple approach which, given distributed computing resources, can nearly achieve the accuracy of k-NN prediction, while maintaining (or beating) the faster prediction time of 1-NN. The approach consists of aggregating _denoised_ 1-NN predictors over *a small number* of distributed subsamples. In the paper we show, both theoretically and experimentally, that small subsample sizes suffice to attain similar performance as k-NN, without sacrificing the computational efficiency of 1-NN. 

  This package contains three classes, which all executes the same function and differs only in application scenarios:
  1. *subNN.py* is our standard parallel approach which runs best in multi-core single computer. It generate sub-models and make predictions in parallelly. *( Be careful that this approach is very memory-consuming as it generate seperate a data structures for each sub-model in RAM. )*
  2. *subNN_seq.py* is a sequential version of *subNN* which runs best in situations where memory is less abundant. Both training data and testing signals are needed at the initialization of this approach. To save memory space, it will one-by-one fit a sub-model on a subsample of training data, use it to predict on testing signals, delete the sub-model. Predictions on testing signals will be summed in the end.
  3. *subNN_remote.py* provides methods for fitting sub-models separately and methods for combination so users can build distributed versions of subNN model upon it. k value should be given by users in this model.

### Usage & Demo

*subNN*
| Function Name | Input | Output | Utility |
|:-:|:-:|:-:|:-:|
| \_\_init\_\_(self, k\_train=10, n_model=10, subSampleRatio=0.1, algorithm_NN='kd_tree')  | *k_train* is the k used in denoising; *n_model* is number of submodels; *subSampleRatio* is the size of subsample used in the submodel; *algorithm_NN* is the algorithm used in finding nearest neighbors, choices are ['brute', 'kd_tree', 'ball_tree']    | void      | initialize
| fit(self, X, y, rSeed=1234) | *X* is training signal matrix of size N*M;  *y* is training target vector of length N; N is number of samples  | void      | fit training data to model
| predict(self, T, parallel=True) | *T*, matrix of size s*M, is new signals to be predicted, s is the number of new data points; *parallel* is True if run the program in parallel | vector of length s, the predicted label of each point | predict the label of new data
| search_k(self, X, y, n_folds=2, n_repeat=1) | *X* is training signal matrix of size N*M;  *y* is training target vector of length N; *n_folds* and *n_repeat* specifies the number of folds and repeatition in cross validation for a given *k* | the *k* giving best cross validation error rate | find the best *k* in training data |

*subNN_seq*
| Function Name | Input | Output | Utility |
|:-:|:-:|:-:|:-:|
| \_\_init\_\_(self, k\_train=10, n_model=10, subSampleRatio=0.1, algorithm_NN='kd_tree')  | *k_train* is the k used in denoising; *n_model* is number of submodels; *subSampleRatio* is the size of subsample used in the submodel; *algorithm_NN* is the algorithm used in finding nearest neighbors, choices are ['brute', 'kd_tree', 'ball_tree']    | void      | initialize
| fit(self, X, y, rSeed=1234) | *X* is training signal matrix of size N*M;  *y* is training target vector of length N; N is number of samples  | void      | fit training data to model
| predict(self, T) | *T*, matrix of size s*M, is new signals to be predicted, s is the number of new data points | vector of length s, the predicted label of each point | predict the label of new data
| search_k(self, X, y, n_folds=2, n_repeat=1) | *X* is training signal matrix of size N*M;  *y* is training target vector of length N; *n_folds* and *n_repeat* specifies the number of folds and repeatition in cross validation for a given *k* | the *k* giving best cross validation error rate | find the best *k* in training data |

*subNN_remote*
| Function Name | Input | Output | Utility |
|:-:|:-:|:-:|:-:|
| \_\_init\_\_(self, k\_train=10, n_model=10, subSampleRatio=0.1, algorithm_NN='kd_tree')  | *k_train* is the k used in denoising; *n_model* is number of submodels; *subSampleRatio* is the size of subsample used in the submodel; *algorithm_NN* is the algorithm used in finding nearest neighbors, choices are ['brute', 'kd_tree', 'ball_tree']    | void      | initialize
| fit(self, X, y, rSeed=1234) | *X* is training signal matrix of size N*M;  *y* is training target vector of length N; N is number of samples  | void      | fit training data to model
| predict(self, T) | *T*, matrix of size s*M, is new signals to be predicted, s is the number of new data points | vector of length s, the predicted label of each point | predict the label of new data
| predict_raw(self, T) | *T*, matrix of size s*M, is new signals to be predicted | *result_list* a list of predictions, each of which is a prediction made by a submodel | 
| predict_combine(self, result_list) | *result_list* is the combine list of outputs of | provide results of submodels that can be combined later *predict_raw* | vector of length s, the predicted label of each point | combine prediction from submodels, submodels can be run in separate remote computers and then combined

To Run Demo:
`
    python3 Demo.py
`


### Reference