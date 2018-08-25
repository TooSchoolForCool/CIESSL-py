import os
import argparse
import random

import numpy as np


class BatchLoader(object):

    def __init__(self, data_dir, mode="all"):
        self.__load_dataset(data_dir, mode)


    def __load_dataset(self, data_dir, mode):
        dataset = []

        for file in os.listdir(data_dir):
            if file.endswith(".pickle"):
                data = np.load( os.path.join(data_dir, file) )
                dataset.append(data)

        if mode == "train":
            idx = np.arange( len(dataset) )
            rs = np.random.RandomState(0)
            rs.shuffle(idx)
            dataset = np.asarray(dataset)[idx[:20]]

        self.dataset_ = dataset


    def size(self):
        return len( self.dataset_ )


    def load_batch(self, batch_size, suffle=True, flatten=True):
        idx = np.arange( len(self.dataset_) )

        if suffle:
            rs = np.random.RandomState( random.randint(0, 2 ** 32 - 1) )
            rs.shuffle(idx)
        
        batch_data = []
        for i in idx:
            data = self.dataset_[i]
    
            if flatten:
                data = data.T.flatten()

            batch_data.append(data)

            if len(batch_data) == batch_size:
                # (n_samples, n_features)
                yield np.asarray(batch_data)
                batch_data = []

        if batch_data:
            yield np.asarray(batch_data)