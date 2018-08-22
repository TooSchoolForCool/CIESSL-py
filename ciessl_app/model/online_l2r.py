from collections import deque

import numpy as np


class OnlineL2R(object):

    def __init__(self, lm, q_size=100):
        """
        Constructor

        Args:
            lm (learning model object)
            q_size (int): max queue size for memorizing previous samples
        """
        self.xq_ = deque(maxlen=q_size)
        self.yq_ = deque(maxlen=q_size)
        self.lm_ = lm


    def partial_fit(self, X, y, classes=None):
        X, y = self.__prep_training_set(X, y)

        if classes is None:
            self.lm_.partial_fit(X, y)
        else:
            self.lm_.partial_fit(X, y, classes=classes)

        
    def predict_proba(self, X):
        return self.lm_.predict_proba(X)


    def __prep_training_set(self, X, y):
        for x_ in X:
            self.xq_.append(x_)
        for y_ in y:
            self.yq_.append(y_)

        X = np.asarray( list(self.xq_) )
        y = np.asarray( list(self.yq_) )

        return X, y