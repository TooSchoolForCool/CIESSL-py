import itertools

import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from scipy import stats


class RankSVM(SGDClassifier):
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


if __name__=="__main__":
    rs = np.random.RandomState(0)
    n_samples_1 = 1000
    n_samples_2 = 300
    X = np.r_[1.5 * rs.randn(n_samples_1, 2),
              0.5 * rs.randn(n_samples_2, 2) + [2, 2]]
    y = np.array([0] * (n_samples_1) + [1] * (n_samples_2))
    idx = np.arange(y.shape[0])
    rs.shuffle(idx)
    X = X[idx]
    y = y[idx]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    
    clf = RankSVM(max_iter=100, alpha=0.01, loss='hinge')
    clf.fit(X[:2], y[:2])

    pred = clf.predict(X)

    print "ACC: %.4f" % metrics.accuracy_score(y, pred)
    print("AUC: %.4f" % metrics.roc_auc_score(y, pred))
    print("Precision: %.4lf" % metrics.average_precision_score(y, pred))
    print("Recall: %.4lf" % metrics.recall_score(y, pred))
    print "CONFUSION MATRIX: "
    print metrics.confusion_matrix(y, pred)
    print "Kendall Tau: %.4f" % kendalltau(clf,X,y)
    print 80*'='

    for i in [x * 10 for x in range(1, (n_samples_1 + n_samples_2) / 10)]:
        clf.partial_fit(X[i:i + 10], y[i:i + 10])

    pred = clf.predict(X)

    print "ACC: %.4f" % metrics.accuracy_score(y, pred)
    print("AUC: %.4f" % metrics.roc_auc_score(y, pred))
    print("Precision: %.4lf" % metrics.average_precision_score(y, pred))
    print("Recall: %.4lf" % metrics.recall_score(y, pred))
    print "CONFUSION MATRIX: "
    print metrics.confusion_matrix(y, pred)
    print "Kendall Tau: %.4f" % kendalltau(clf,X,y)
    print 80*'='
