import pickle
import argparse
import os
import sys
import random

import numpy as np
from skmultilearn.adapt import MLARAM
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from model.online_clf import OnlineClassifier
from model.evaluator import Evaluator
import utils


def arg_parser():
    """Argument Parser

    Parse arguments from command line, and perform error checking

    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='Training Voice AutoEncoder')

    parser.add_argument(
        "--data",
        dest="data",
        type=str,
        required=True,
        help="Input data directory"
    )
    parser.add_argument(
        "--map",
        dest="map",
        type=str,
        required=True,
        help="Map feature directory"
    )
    
    args = parser.parse_args()
    return args


class CNN(nn.Module):
    def __init__(self, out_size=3):
        super(CNN, self).__init__()

        self.nn_ = nn.Sequential(
            nn.Conv2d(120, 240, (3, 3), stride=(2, 2)),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(240, 480, (3, 3), stride=(2, 2)),
            # # nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(480, 960, (3, 3), stride=(2, 2)),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(960, 480, (3, 3), stride=(1, 1)),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(480, 3, (2, 2), stride=(1, 1)),
            # nn.BatchNorm2d(256),
            nn.ReLU(True)

            # nn.Conv2d(1024, code_size, (1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(code_size),
        )


    def partial_fit(self, X, y):
        X = torch.Tensor(X)
        X = Variable(X)
        if torch.cuda.is_available():
            X = X.cuda()

        y = torch.Tensor(y)
        y = Variable(y)
        if torch.cuda.is_available():
            y = y.cuda()

        out = self.nn_(X)
        out = out.view(out.size(0), -1)
        print(out.shape)
        print(y.shape)
        exit(0)
        # loss = model.loss(recon_batch, data)
        # # backward
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # train_loss += loss.item()


class Loader(object):
    def __init__(self, data_dir, map_dir=None):
        self.dataset_ = self.load_dataset_(data_dir)
        self.map_ = np.load(map_dir)


    def load_dataset_(self, data_dir):
        files = []
        for file in os.listdir(data_dir):
            if file.endswith(".pickle"):
                files.append(os.path.join(data_dir, file))

        dataset = []
        for file in files:
            with open(file, "rb") as file_in:
                data = pickle.load(file_in)
            dataset.append( data )

        return dataset

    def calc_map_feature(self):
        map_feature = np.copy(self.map_)
        entropy = np.random.rand(1, map_feature.shape[0]) * 0.05
        map_feature += entropy[0]

        return map_feature


    def init_training_set(self, seed=0):
        idx = np.arange( len(self.dataset_) )
        rs = np.random.RandomState(seed)
        rs.shuffle(idx)

        map_feature = self.calc_map_feature()
        return self.dataset_[idx[0]]["gccfb"], self.dataset_[idx[0]]["room_id"]


    def data_iterator(self, n_samples=None, seed=0, shuffle=True):
        """
        Yield voice data one by one interatively

        Args:
            n_samples (int): number of total samples to be yield
            seed (int): random seed for the numpy shuffling
            shuffle (bool): shuffle data or not

        Yields:
            data (dictionary):
                data["gccfb"] (n_mels, n_gcc): gccfb feature
                data["room_id"] (int): room id 
        """
        idx = np.arange( len(self.dataset_) )
        n_mels = 40

        if shuffle:
            rs = np.random.RandomState(seed)
            rs.shuffle(idx)

        n_samples = len(self.dataset_) if n_samples is None else n_samples

        for i in idx[:n_samples]:
            map_feature = self.calc_map_feature()
            gccfb_feature = self.dataset_[i]["gccfb"][:, :n_mels, :20].flatten()

            X = np.asarray( [np.append(gccfb_feature, map_feature)] )
            
            yield X, self.dataset_[i]["room_id"]


def train_gccfb():
    args = arg_parser()

    n_rooms = 3
    loader = Loader(args.data, args.map)
    clf = MLARAM(vigilance=0.5, threshold=0.1)
    l2r = OnlineClassifier(clf, q_size=1, shuffle=True)

    for X, y in loader.data_iterator(seed=random.randint(0, 1000)):
        rank_y = utils.label2rank(label=y, n_labels=n_rooms)
        l2r.fit(X, rank_y)
        break

    evaluator = Evaluator(n_rooms, verbose=True)
    for X, y in loader.data_iterator(seed=random.randint(0, 1000)):
        rank_y = utils.label2rank(label=y, n_labels=n_rooms)

        predicted_y = l2r.predict_proba(X)
        evaluator.evaluate([y], predicted_y)

        l2r.fit(X, rank_y)


if __name__ == '__main__':
    train_gccfb()