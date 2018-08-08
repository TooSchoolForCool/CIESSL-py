import random
import numpy as np
import os
import sys


class DataLoader(object):
    """
    Handle data loading and parsing.
    """

    def __init__(self, voice_data_dir, map_data_dir):
        """
        Store voice dataset directory and map data directory

        Args:
            voice_data_dir (string): voice dataset directory where
                multiple sound track files (.wav) are stored
            map_data_dir (string): a specific file path to a .json file
                which stores the map information
        """
        self.map_file_dir_ = map_data_dir

        # load all files' directories in voice_data_dir
        self.voice_file_dirs_ = []
        for file in os.listdir(voice_data_dir):
            self.voice_file_dirs_.append(os.path.join(voice_data_dir, file))


    def voice_data_iterator(self, n_samples=None, seed=0):
        rs = np.random.RandomState(seed)
        idx = np.arange( len(self.voice_file_dirs_) )
        rs.shuffle(idx)

        n_samples = len(self.voice_file_dirs_) if n_samples is None else n_samples

        for i in idx[:n_samples]:
            voice_frames = np.load(self.voice_file_dirs_[i])

            yield voice_frames


def test():
    voice_data_dir = "../../data/active_voice"
    map_data_dir = "../../data/map"

    data_loader = DataLoader(voice_data_dir, map_data_dir)
    
    cnt = 0
    for frame in data_loader.voice_data_iterator(n_samples=100, seed=0):
        print("frame #%d: %r" % (cnt, frame.shape))
        cnt += 1


if __name__ == '__main__':
    test()