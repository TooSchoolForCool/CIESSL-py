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


    def voice_data_iterator(self, n_samples=None, seed=0, shuffle=True):
        """
        Yield voice data one by one interatively

        Args:
            n_samples (int): number of total samples to be yield
            seed (int): random seed for the numpy shuffling
            shuffle (bool): shuffle data or not

        Yields:
            data (dictionary):
                data["samplerate"] (int): samplerate of the voice data
                data["src"] (int): source index (where sound source is placed)
                data["dst"] (int): destination index (where mic array is placed)
                data["room"] (int): room index (in which room mic array is placed)
                data["idx"] (int): sample index (the i-th sample that in <src, dst> dataset)
                data["frames"] ( np.ndarray (n_samples, n_channels) ): 
                    sound signal frames from every mic channel
        """
        idx = np.arange( len(self.voice_file_dirs_) )

        if shuffle:
            rs = np.random.RandomState(seed)
            rs.shuffle(idx)

        n_samples = len(self.voice_file_dirs_) if n_samples is None else n_samples

        for i in idx[:n_samples]:
            voice_frames = np.load(self.voice_file_dirs_[i])

            data = self.__parse_voice_filename(self.voice_file_dirs_[i])
            data["frames"] = voice_frames

            yield data


    def __parse_voice_filename(self, file_dir):
        """
        This function is used to parse the filename of the voice data,
        and return its relavent information.

        Args:
            file_dir (string): directory to the file

        Returns:
            info (dictionary):
                # info['filename']: filename with directory prefix removed (/a/b/123.wav --> 123)
                info['samplerate'] (int): samplerate of the voice data
                info['src'] (int): source index (where sound source is placed)
                info['dst'] (int): destination index (where mic array is placed)
                info['room'] (int): room index (in which room mic array is placed)
                info['idx'] (int): sample index (the i-th sample that in <src, dst> dataset)
        """
        filename = file_dir.split('/')[-1].split('.')[0]
        filename_split = filename.split('-')

        samplerate = int(filename_split[0])
        src = int(filename_split[1])
        dst = int(filename_split[2])
        room = int(filename_split[3])
        idx = int(filename_split[4])

        info = {}
        # info["filename"] = filename
        info["samplerate"] = samplerate
        info["src"] = src
        info["dst"] = dst
        info["room"] = room
        info["idx"] = idx

        return info


def test():
    voice_data_dir = "../../data/active_voice"
    map_data_dir = "../../data/map"

    data_loader = DataLoader(voice_data_dir, map_data_dir)
    
    cnt = 0
    for data in data_loader.voice_data_iterator(n_samples=100, seed=0):
        print("frame #%d: %r" % (cnt, data["frames"].shape))
        cnt += 1


if __name__ == '__main__':
    test()