import numpy as np
import matplotlib.pyplot as plt


class RankCLF(object):
    def __init__(self, n_classes, kernel="rbf", C=1.0, n_iter=1):
        self.n_classes_ = n_classes

        if kernel == "rbf":
            self.kernel_ = self.rbf_kernel_

        self.C_ = C
        self.n_iter_ = n_iter

        # ranking function params
        self.coef_ = None
        self.train_X_ = None
        self.train_y_ = None


    def rbf_kernel_(self, x1, x2):
        sigma = 10.0
        dist = np.linalg.norm(x1 - x2, 2) ** 2
        rbf = np.exp( -(dist / (2 * sigma)) )
        # print(x1)
        # print(x2)
        # print(dist)
        # print(rbf)
        return rbf


    def fit(self, X, y):
        n_samples = X.shape[0]
        K = self.n_classes_
        C = self.C_
        n_iter = self.n_iter_

        # create kernel matrix
        k_mat = self.create_kernel_mat_(X)

        # create alpha matrix (n_samples, n_classes)
        alpha = [[0.1 * C for _ in range(K)] for _ in range(n_samples)]
        alpha = np.asarray(alpha)

        for iter in range(0, n_iter):
            for i in range(0, n_samples):
                # print("sample {}".format(i))
                _lambda = self.calc_lambda_(alpha, X, y, i, k_mat)
                # update alpha vector for sample i
                for k in range(0, K):
                    new_val = (1 + _lambda*y[i, k] - 0.5 * y[i, k] \
                        * self.leave_one_out_predict_(alpha, X, y, i, k, k_mat)) / k_mat[i, i]
                    alpha[i, k] = self.project_func_(0, C, new_val)

        self.coef_ = alpha
        self.train_X_ = X
        self.train_y_ = y

        # print("training is done: {}".format(self.coef_))
        

    def predict(self, X):
        labels = []
        K = self.n_classes_

        for x in X:
            rank = self.ranking_score(x)
            idx = np.argmax(rank)
            label = [1 if k == idx else -1 for k in range(0, K)]
            labels.append(label)

        return labels



    def predict_proba(self, X):
        y_proba = []

        for x in X:
            rank = self.ranking_score(x)
            y_proba.append(rank)

        return np.asarray(y_proba)


    def calc_lambda_(self, alpha, X, y, i, k_mat):
        if(self.calc_g_lambda_(alpha, X, y, i, 0, k_mat) >= 0):
            a = -999.0
            b = 0.0
        if(self.calc_g_lambda_(alpha, X, y, i, 0, k_mat) <= 0):
            a = 0.0
            b = 999.0

        assert(self.calc_g_lambda_(alpha, X, y, i, a, k_mat) <= 0)
        assert(self.calc_g_lambda_(alpha, X, y, i, b, k_mat) >= 0)

        max_iter = 1000
        TOL = 1e-5

        for _ in range(max_iter):
            c = (a + b) / 2
            f_val = self.calc_g_lambda_(alpha, X, y, i, c, k_mat)

            if f_val == 0:
                return c
            if (b - a) / 2 < TOL:
                return c

            if f_val < 0:
                a = c
            else:
                b = c

        return (a + b) / 2


    def calc_g_lambda_(self, alpha, X, y, i, _lambda, k_mat):
        K = self.n_classes_
        C = self.C_

        val = 0.0
        for k in range(0, K):
            hx = (y[i, k] + _lambda - 0.5 * self.leave_one_out_predict_(alpha, X, y, i, k, k_mat)) \
                / k_mat[i, i]
            hy = y[i, k] * C

            val += self.h_func_(hx, hy)

        return val


    def h_func_(self, x, y):
        if y > 0:
            return self.project_func_(0, y, x)
        else:
            return self.project_func_(y, 0, x)


    def project_func_(self, a, b, x):
        if x > b:
            return b
        elif x < a:
            return a
        else:
            return x


    def sum_row_(self, M, row):
        return np.sum(M[row, :])


    def create_kernel_mat_(self, X):
        n = X.shape[0]
        k_mat = np.zeros((n, n))
        
        for i in range(0, n):
            for j in range(i, n):
                k_mat[j, i] = k_mat[i, j] = self.kernel_(X[i], X[j])

        return k_mat


    def ranking_score(self, x):
        X = self.train_X_
        y = self.train_y_
        alpha = self.coef_
        K = self.n_classes_
        n = X.shape[0]

        rank = []

        for k in range(0, K):
            score = 0.0
            for i in range(0, n):
                score += y[i, k] * alpha[i, k] * self.kernel_(X[i], x)
            rank.append(score)

        return np.asarray(rank)


    def leave_one_out_predict_(self, alpha, X, y, i, k, k_mat):
        """
        Args:
            alpha nd.array(n_samples, n_classes): alpha matrix
            X (n_samples, n_features): training feature set
            y binary matrix (n_samples, n_classes): training label set
            i (int): target index
            k (int): label index [0, n_classes]
        """
        score = 0.0
        n = X.shape[0]

        for j in range(0, n):
            if i == j:
                continue
            score += k_mat[j, i] * alpha[j, k] * y[j, k]

        return score


def read_data_set(file_path, n_samples=None):
    file = open(file_path, "r")

    X, y = [], []
    for line in file:
        if line is None:
            break
        data = line[:-2].split('\t')

        X.append(data[:-1])
        y.append(data[-1])

    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int")

    idx = np.arange( len(X) )
    rs = np.random.RandomState(0)
    rs.shuffle(idx)

    if n_samples is not None:
        idx = idx[:n_samples]

    return X[idx], y[idx]


def test_cvx():
    data_path = "dataset.txt"
    X, y = read_data_set(data_path)

    # reshape label y to multi-label format
    y = [[1 if i == yi else -1 for i in range(1, 4)] for yi in y]
    y = np.asarray(y)

    clf = RankCLF(n_classes=3)
    clf.fit(X[:10], y[:10])

    acc = 0
    for xi, yi in zip(X[10:], y[10:]):
        pred_y = clf.predict([xi])

        print("y:\t{}".format(yi))
        print("pred_y:\t{}".format(pred_y[0]))

        if list(pred_y[0]) == list(yi):
            acc += 1
            print("hit: {}".format(acc))

    print("acc: {}".format(1.0 * acc / (X.shape[0] - 10)))


if __name__ == '__main__':
    test_cvx()