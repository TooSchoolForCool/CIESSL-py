import numpy as np
import matplotlib.pyplot as plt


class RankFOGD(object):
    def __init__(self, n_classes, loss="rank_hinge", C=1.0, D=1000, eta=1e-3, sigma=1.0):
        # number of types of classes
        self.n_classes_ = n_classes
        # relax coef in objective function
        self.C_ = C
        # Fourier components size
        self.D_ = D
        # learning rate
        self.eta_ = eta
        # rbf (gaussian) kernel width
        self.sigma_ = sigma

        if loss == "hinge":
            self.update_weights_ = self.hinge_loss_update_
        elif loss == "rank_hinge":
            self.update_weights_ = self.rank_loss_update_

        # approx weight vector with shape (n_classes, D * 2)
        self.w_ = None
        # Fourier Components with shape (n_features, D)
        self.u_ = None


    def partial_fit(self, X, y):
        if self.w_ is None:
            self.w_ = np.zeros((self.n_classes_, self.D_*2))
        if self.u_ is None:
            self.u_ = (1.0 / self.sigma_) * np.random.randn(X.shape[1], self.D_)

        w = self.w_
        u = self.u_

        for x_, y_ in zip(X, y):
            assert(y_.shape[0] == self.n_classes_)
            assert(x_.shape[0] == u.shape[0])

            zx = self.fourier_repr_(x_, u)

            # calculate score for each class
            score = self.calc_score_(w, zx)
            # update weight vector
            w = self.update_weights_(w, score, y_, zx)

        # save weights and Fourier Components
        self.w_ = w
        self.u_ = u


    def predict(self, X):
        K = self.n_classes_
        labels = []

        for x in X:
            zx = self.fourier_repr_(x, self.u_)
            score = self.calc_score_(self.w_, zx)

            idx = np.argmax(score)
            label = [1 if k == idx else -1 for k in range(0, K)]
            labels.append(label)

        return labels


    def predict_proba(self, X):
        y_proba = []

        for x in X:
            zx = self.fourier_repr_(x, self.u_)
            rank_score = self.calc_score_(self.w_, zx)
            y_proba.append(rank_score)

        return np.asarray(y_proba)


    def fourier_repr_(self, x, u):
        # transform feature to Fourier space
        cos_ux = np.cos(np.dot(u.T, x))
        sin_ux = np.sin(np.dot(u.T, x))
        # shape: (2 * D, )
        zx = np.append(cos_ux, sin_ux)

        return zx


    def rank_loss_update_(self, w, score, y, zx):
        K = self.n_classes_

        for k in range(0, K):
            for l in range(k, K):
                if y[k] == y[l]:
                    continue

                delta_kl = 0.5 * (y[k] - y[l]) * (score[k] - score[l])
                
                # gradient max(0, 1 - delta_kl)
                if delta_kl < 1:
                    w[k] = w[k] + self.eta_ * 0.5*(y[k] - y[l]) * zx
                    w[l] = w[l] - self.eta_ * 0.5*(y[k] - y[l]) * zx

        return w


    def hinge_loss_update_(self, w, score, y, zx):
        yt = np.argmax(y)
        # exclude yt index
        rest = [i for i in range(0, yt)] + [i for i in range(yt+1, self.n_classes_)]
        st = np.argmax(score[rest])

        loss = max( 0, 1 - (score[yt] - score[st]) )

        if loss > 0:
            w[yt] = w[yt] + self.eta_ * zx
            w[st] = w[st] - self.eta_ * zx

        return w


    def calc_score_(self, w, zx):
        return np.dot(w, zx)
        


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

    clf = RankFOGD(n_classes=3, eta=5e-4)
    clf.partial_fit(X[:5], y[:5])

    acc = 0
    for xi, yi in zip(X[5:], y[5:]):
        pred_y = clf.predict([xi])

        print("y:\t{}".format(yi))
        print("pred_y:\t{}".format(pred_y[0]))

        if list(pred_y[0]) == list(yi):
            acc += 1
            print("hit: {}".format(acc))

        clf.partial_fit([xi], [yi])

    print("acc: {}".format(1.0 * acc / (X.shape[0] - 5)))


if __name__ == '__main__':
    test_cvx()