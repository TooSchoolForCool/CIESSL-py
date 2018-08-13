import itertools

import numpy as np

from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn import metrics
from scipy import stats


class RankSVM(SGDRegressor):
    """Performs pointwise ranking with an underlying SGDClassifer model
    """

    def fit(self, X, y):
        """
        Fit a ranking model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,)
        """
        super(RankSVM, self).fit(X, y)


    def partial_fit(self, X, y):
        """
        Perform online learning with SGD

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,)
        """
        super(RankSVM, self).partial_fit(X, y)


    def predict(self, X):
        """
        Perform prediction on the input data

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns:
        pred ( np.ndarray, shape (n_samples,) ): prediction result
        """
        pred = super(RankSVM, self).predict(X)
        return pred


def kendalltau(clf,X,y):
    if clf.coef_.shape[0] == 1:
        coef = clf.coef_[0]
    else:
        coef = clf.coef_     
    tau, _ = stats.kendalltau(np.dot(X, coef), y)
    return np.abs(tau)



def regressor_test():
    from sklearn.metrics import mean_absolute_error

    rs = np.random.RandomState(0)
    guassian = lambda : 1 * (rs.randn()) + 1
    func = lambda x : (3.0 * x)

    n_samples = 25
    X = [[i] for i in range(0, n_samples)]
    y = [func(t[0]) for t in X]

    X = np.asarray(X)
    y = np.asarray(y)

    idx = np.arange(y.shape[0])
    rs.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # mean = X.mean(axis=0)
    # std = X.std(axis=0)
    # X = (X - mean) / std
    
    clf = RankSVM(max_iter=100, alpha=0.01)
    # clf.fit(X, y)
    clf.partial_fit(X, y)
    
    # step = 1500
    # for i in range(0, 2500, step):
    #     # pred = clf.predict(X[i:i+1])
    #     # print(mean_absolute_error(y[i:i+1], pred))

    #     clf.partial_fit(X[i : i+step], y[i : i+step])
    #     break


    pred = clf.predict(X[:])
    print("Final result: ", mean_absolute_error(y, pred))


if __name__=="__main__":
    regressor_test()
