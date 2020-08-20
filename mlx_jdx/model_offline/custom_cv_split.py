import numpy as np
import numbers
from matplotlib.cbook import flatten
from sklearn.model_selection import BaseCrossValidator


class TimeWindowSplit(BaseCrossValidator):
    '''
    split data as time.
    divide data into 2*n_splits parts;
    n_splits parts as the train set and the next part as validation set,
    e.g. 6 parts and their index [0,1,2,3,4,5].

    if fix_window = True
    train [0,1,2] test[3]
    train [1,2,3] test[4]
    train [2,3,4] test[5]

    if fix_window = False
    train [0,1,2] test[3]
    train [0,1,2,3] test[4]
    train [0,1,2,3,4] test[5]

    '''

    def __init__(self, n_splits, fix_window=True):
        super(TimeWindowSplit,self).__init__()

        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))

        if not isinstance(fix_window, bool):
            raise TypeError("fix_window must be True or False;"
                            " got {0}".format(fix_window))

        self.n_splits = n_splits
        self.fix_window = fix_window

    def split(self, X, y=None, groups=None):
        X = np.array(X)
        partition = 2 * self.n_splits
        #ensure partition is smaller than the input split length
        if X.shape[0] < partition:
            partition = X.shape[0]
        partial_X_row = np.array_split(range(X.shape[0]), partition, axis=0)
        for ii in range(self.n_splits, partition):
            if self.fix_window:
                train_index = list(flatten(map(lambda x: partial_X_row[x], range(ii - self.n_splits, ii))))
            else:
                train_index = list(flatten(map(lambda x: partial_X_row[x], range(ii))))
            test_index = partial_X_row[ii]

            yield np.array(train_index,dtype=np.int64), np.array(test_index,dtype=np.int64)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        '''This function must be implemented, but it is available to set 'pass'
        '''
        X = np.array(X)
        partition = 2 * self.n_splits
        partial_X_row = np.array_split(range(X.shape[0]), partition, axis=0)
        for jj in range(self.n_splits, partition):
            yield partial_X_row[jj]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits