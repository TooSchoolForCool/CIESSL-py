import pickle
import argparse
import os
import sys
import random
import Queue
import json

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


class Loader(object):
    def __init__(self, data_dir, map_dir=None):
        self.dataset_ = self.load_dataset_(data_dir)
        self.parse_map_(map_dir)


    def parse_map_(self, map_dir):
        json_file=open(map_dir).read()
        data = json.loads(json_file)

        self.n_rooms_ = data["n_rooms"]
        self.room_centers_ = [(c["x"], c["y"]) for c in data["room_centers"]]
        # to access the (x, y) point in the map, use map[y, x]
        self.segmented_map_ = np.asarray(data["map"], dtype=np.int32)
        self.resolution_ = data["resolution"]
        self.origin_ = (-int(data["origin"]["x"] / data["resolution"]), 
            -int(data["origin"]["y"] / data["resolution"]))
        
        bl = (data["boundary"]["min_x"], data["boundary"]["min_y"])
        tr = (data["boundary"]["max_x"], data["boundary"]["max_y"])
        self.boundary_ = {"bl" : bl, "tr" : tr}


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
        mic_flooding_map = self.__flooding_map(self.segmented_map_, self.room_centers_[0],
            self.boundary_, 0.999)
        shrink_map = self.__shrink_map(mic_flooding_map, kernel_size=(25, 25))
        map_feature = shrink_map.flatten().astype(np.float32)

        return map_feature


    def __shrink_map(self, img, kernel_size=(3, 3)):
        # reshape image in terms of pytorch conv requirements
        img = img.astype(np.float32)
        img = torch.from_numpy(img.reshape((1, 1, img.shape[0], img.shape[1])))
        img = F.avg_pool2d(Variable(img), kernel_size=kernel_size)
        img = img.data.squeeze().numpy().astype("float32")
        return img


    def __flooding_map(self, grid_map, center, boundary, rate):
        flooding_map = np.zeros((grid_map.shape[0], grid_map.shape[1], 1), np.float32)
        visited = np.zeros(grid_map.shape, np.uint8)

        pivot = 1.0
        height, width = grid_map.shape

        q = Queue.Queue()
        q.put( (center[0], center[1]) )
        visited[center[1], center[0]] = 1

        while not q.empty():
            level_size = q.qsize()
            for _ in range(0, level_size):
                x, y = q.get()

                if grid_map[y, x] == 0:
                    continue
                
                if x + 1 < width and visited[y, x + 1] == 0:
                    q.put( (x + 1, y) )
                    visited[y, x + 1] = 1
                if x - 1 >= 0 and visited[y, x - 1] == 0:
                    q.put( (x - 1, y) )
                    visited[y, x - 1] = 1
                if y + 1 < height and visited[y + 1, x] == 0:
                    q.put( (x, y + 1) )
                    visited[y + 1, x] = 1
                if y - 1 >= 0 and visited[y - 1, x] == 0:
                    q.put( (x, y - 1) )
                    visited[y - 1, x] = 1

                flooding_map[y, x] = pivot

            entropy = min(0.0524 * np.random.rand(1)[0] + 0.97, rate)
            # print("entropy: {}".format(entropy))
            pivot *= entropy

        # crop out map region
        min_x, min_y = boundary["bl"][0], boundary["bl"][1]
        max_x, max_y = boundary["tr"][0], boundary["tr"][1]

        flooding_map = flooding_map[min_y:max_y, min_x:max_x]
        # flooding_map = cv2.GaussianBlur(flooding_map, (3, 3), 0.01);

        return flooding_map


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
        n_mels = 30

        if shuffle:
            rs = np.random.RandomState(seed)
            rs.shuffle(idx)

        n_samples = len(self.dataset_) if n_samples is None else n_samples

        for i in idx[:n_samples]:
            map_feature = self.calc_map_feature()
            gccfb_feature = self.dataset_[i]["gccfb"][:, :n_mels, :].flatten()

            X = np.asarray( [np.append(gccfb_feature, map_feature)] )
            
            yield X, self.dataset_[i]["room_id"]


def train_gccfb():
    args = arg_parser()

    n_rooms = 3
    n_trails = 100
    eval_out_dir = "HARAM-GCCFB"
    loader = Loader(args.data, args.map)


    for t in range(0, n_trails):
        clf = MLARAM(vigilance=0.999, threshold=0.02)
        l2r = OnlineClassifier(clf, q_size=1, shuffle=True)
        evaluator = Evaluator(n_rooms, verbose=True)

        for X, y in loader.data_iterator(seed=random.randint(0, 1000)):
            rank_y = utils.label2rank(label=y, n_labels=n_rooms)
            l2r.fit(X, rank_y)
            break
        
        for X, y in loader.data_iterator(seed=random.randint(0, 1000)):
            rank_y = utils.label2rank(label=y, n_labels=n_rooms)

            predicted_y = l2r.predict_proba(X)
            evaluator.evaluate([y], predicted_y)

            l2r.fit(X, rank_y)
            evaluator.save_history(out_dir=eval_out_dir, file_prefix=str(t), type="csv")


if __name__ == '__main__':
    train_gccfb()


"""
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
"""