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


    def load_data(self, n_samples, )
        rs = np.random.RandomState(0)



def test():
    voice_data_dir = "../../data/sample"
    map_data_dir = "../../data/map"

    data_loader = DataLoader(voice_data_dir, map_data_dir)




if __name__ == '__main__':
    test()