import random
import os
import sys
import json

import numpy as np
import cv2

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
        # load all files' directories in voice_data_dir
        self.voice_file_dirs_ = []
        for file in os.listdir(voice_data_dir):
            self.voice_file_dirs_.append(os.path.join(voice_data_dir, file))

        # parse map data
        json_file=open(map_data_dir).read()
        data = json.loads(json_file)

        self.n_rooms_ = data["n_rooms"]
        self.room_centers_ = [(c["x"], c["y"]) for c in data["room_centers"]]
        self.segmented_map_ = np.asarray(data["map"], dtype=np.int32)


    def save_segmented_map(self, output_path="segmented_map.png"):
        """
        Save segmented map image to local

        Args:
            output_path (string): output path of the image
        """
        img = np.zeros((self.segmented_map_.shape[0], self.segmented_map_.shape[1], 3), np.uint8)

        # paint different room with random color
        for i in range(1, self.n_rooms_ + 1):
            blue = random.randint(0, 256)
            green = random.randint(0, 256)
            red = random.randint(0, 256)

            # traverse each 
            for x in range(0, self.segmented_map_.shape[0]):
                for y in range(0, self.segmented_map_.shape[1]):
                    if self.segmented_map_[x, y] == i:
                        img[x, y] = (blue, green, red)

        # paint estimated room center
        for c in self.room_centers_:
            cv2.circle(img, c, 2, (255, 0, 0), -1)

        cv2.imwrite(output_path, img)


    def load_map_info(self):
        """
        Load segmented map

        Returns:
            map_data (dictionary):
                map_data["data"] ( np.ndarray (height, width) ): 2D room occupancy grid map,
                    each item is an integer in range(0, N). 0 represents wall, and 1 ~ N 
                    represents room index
                map_data["n_room"] (int): number of rooms
                map_data["center"] ( list of 2-item tuple (x, y) ): each 2-item tuple
                    represents a center of a room
        """
        map_data = {}

        map_data["data"] = self.segmented_map_
        map_data["n_room"] = self.n_rooms_
        map_data["center"] = self.room_centers_

        return map_data


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
    map_data_dir = "../../data/map/bh9f_lab_map.json"

    data_loader = DataLoader(voice_data_dir, map_data_dir)
    data_loader.save_segmented_map()

    cnt = 0
    for data in data_loader.voice_data_iterator(n_samples=100, seed=0):
        print("frame #%d: %r" % (cnt, data["frames"].shape))
        cnt += 1


if __name__ == '__main__':
    test()