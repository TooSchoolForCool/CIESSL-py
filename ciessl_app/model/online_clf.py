import random
from collections import deque

import numpy as np


class OnlineClassifier(object):

    def __init__(self, lm, q_size=50, shuffle=False):
        """
        Constructor

        Args:
            lm (learning model object)
            q_size (int): max queue size for memorizing previous samples
            shuffle (bool): shuffle the training dataset or not
        """
        self.xq_ = deque(maxlen=q_size)
        self.yq_ = deque(maxlen=q_size)
        self.lm_ = lm
        self.shuffle_ = shuffle


    def fit(self, X, y):
        self.lm_.fit(X, y)


    def partial_fit(self, X, y, classes=None, n_iter=10):
        # add new-coming data into memory
        for x_ in X:
            self.xq_.append(x_)
        for y_ in y:
            self.yq_.append(y_)

        # initialize partial fit
        if classes is not None:
            self.lm_.partial_fit(X, y, classes=classes)
            n_iter -= 1  

        # run a number of interations
        for _ in range(0, n_iter):
            X, y = self.__prep_training_set()
            self.lm_.partial_fit(X, y)

        
    def predict_proba(self, X):
        return self.lm_.predict_proba(X)


    def __prep_training_set(self):
        """
        Shuffling dataset
        """
        idx = np.arange( len(self.xq_) )
        if self.shuffle_:
            rs = np.random.RandomState( random.randint(0, 2 ** 32 - 1) )
            rs.shuffle(idx)

        X = np.asarray( list(self.xq_) )[idx]
        y = np.asarray( list(self.yq_) )[idx]

        return X, y