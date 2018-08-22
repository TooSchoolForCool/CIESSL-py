import random
import os
import sys
import json

import numpy as np
import cv2
from sklearn.preprocessing import normalize

sys.path.append(os.path.abspath(os.path.join("..")))
from utils import save_segmented_map

class DataLoader(object):
    """
    Handle data loading and parsing.
    """
    def __init__(self, voice_data_dir, map_data_dir, pos_tf_dir, verbose=False, 
        all_in=True, is_normalize=False):
        """
        Store voice dataset directory and map data directory

        Args:
            voice_data_dir (string): voice dataset directory where
                multiple sound track files (.wav) are stored
            map_data_dir (string): a specific file path to a .json file
                which stores the map information
            pos_tf_dir (string): directory of position tranform config file
            verbose (bool): whether print out verbose info
            all_in (bool): load whole dataset into memory at a time
        """
        # load all files' directories in voice_data_dir
        self.voice_file_dirs_ = []
        self.dataset_ = None
        self.is_normalize_ = is_normalize

        for file in os.listdir(voice_data_dir):
            if file.endswith(".pickle"):
                self.voice_file_dirs_.append(os.path.join(voice_data_dir, file))

        self.__parse_map_data(map_data_dir)
        self.__parse_pos_tf(pos_tf_dir)

        if verbose:
            self.print_src_info()

        if all_in:
            self.load_whole_dataset()


    def save_segmented_map(self, output_path="segmented_map.png"):
        """
        Save segmented map image to local

        Args:
            output_path (string): output path of the image
        """
        save_segmented_map(map_data=self.segmented_map_, n_room=self.n_rooms_, 
            room_centers=self.room_centers_, origin=self.origin_, src=self.src_pos_, 
            dst=self.dst_pos_, boundary=self.boundary_, output_path="segmented_map.png")


    def load_map_info(self):
        """
        Load segmented map

        Returns:
            map_data (dictionary):
                map_data["data"] ( np.ndarray (height, width) ): 2D room occupancy grid map,
                    each item is an integer in range(0, N). 0 represents wall, and 1 ~ N 
                    represents room index
                map_data["n_room"] (int): number of rooms
                map_data["origin"] ( tuple (int, int) ): the index of the origin of the map
                map_data["center"] ( list of tuples (int, int) ): each 2-item tuple
                    represents a center of a room
                map_data["boundary"] (dict): boundary information of the map, bottom-left corner,
                    and the top-right corner
        """
        map_data = {}

        map_data["data"] = self.segmented_map_
        map_data["n_room"] = self.n_rooms_
        map_data["center"] = self.room_centers_
        map_data["origin"] = self.origin_
        map_data["boundary"] = self.boundary_

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
                data["src"] (int, int): coordinate of the sound source in the map
                data["src_idx"] (int): sound source index
                data["dst"] (int, int): coordinate of the microphone in the map
                data["frames"] ( np.ndarray (n_samples, n_channels) ): 
                    sound signal frames from every mic channel
        """
        idx = np.arange( len(self.voice_file_dirs_) )

        if shuffle:
            rs = np.random.RandomState(seed)
            rs.shuffle(idx)

        n_samples = len(self.voice_file_dirs_) if n_samples is None else n_samples

        for i in idx[:n_samples]:
            # load voice pickle file
            # voice is store as a np.ndarray (frames, n_channels)

            if self.dataset_ is None:
                voice_frames = np.load(self.voice_file_dirs_[i])
            else:
                voice_frames = self.dataset_[i]
                if self.is_normalize_:
                    voice_frames = normalize(voice_frames)

            info = self.__parse_voice_filename(self.voice_file_dirs_[i])

            data = {}
            data["frames"] = voice_frames
            data["samplerate"] = info["samplerate"]
            data["src"] = self.src_pos_[info["src"] - 1]
            data["src_idx"] = info["src"]
            data["dst"] = self.dst_pos_[info["dst"] - 1]

            yield data


    def print_src_info(self):
        for i, src in enumerate(self.src_pos_):
            print("src %d: %r" % (i + 1, src))

        for i, dst in enumerate(self.dst_pos_):
            print("dst %d: %r" % (i + 1, dst))


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


    def __parse_map_data(self, map_data_dir):
        """
        Parsing map data
    
        Args:
            map_data_dir (string): map data directory
        """
        # map_data (dictionary):
        #     map_data["data"] ( np.ndarray (height, width) ): 2D room occupancy grid map,
        #         each item is an integer in range(0, N). 0 represents wall, and 1 ~ N 
        #         represents room index
        #     map_data["n_room"] (int): number of rooms
        #     map_data["origin"] ( dict (x, y) ): the origin of the map,
        #         here the origin is not in the index form, but is in the meter form
        #     map_data["resolution"] (double): meter per cell, origin / resolution is the
        #         index of the origin
        #     map_data["center"] ( list of 2-item tuple (x, y) ): each 2-item tuple
        #         represents a center of a room
        json_file=open(map_data_dir).read()
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


    def __parse_pos_tf(self, pos_tf_dir):
        """
        Parsing position transform file

        Args:
            pos_tf_dir (string): position transform file directory
        """
        # parse the position transform file
        # the transform is in meter scale, use map resolution to calculate the index/cell scale
        json_file = open(pos_tf_dir).read()
        pos_tf = json.loads(json_file)
        # real-world origin to simu-world origin transform (delta_x, delta_y)
        # meter scale
        real2simu_tf = pos_tf["real2simu_tf"]
        # A list sound source locations [(x1, y1), ...] corresponding to the real world 
        # origin (manually chosen), meter scale
        src_pos = pos_tf["src"]
        # A list microphone locations [(x1, y1), ...] corresponding to the real world 
        # origin (manually chosen), meter scale
        dst_pos = pos_tf["dst"]

        # convert real-world origin to simu-world origin transform into cell-scale
        delta_x = real2simu_tf["delta_x"] / self.resolution_
        delta_y = real2simu_tf["delta_y"] / self.resolution_
        self.real2simu_tf_ = (int(delta_x), int(delta_y))

        # convert src position to cell-scale (in which cell is the sound source)
        self.src_pos_ = []
        # src 1 start from index 0
        for src in src_pos:
            x = src["x"] / self.resolution_ - self.real2simu_tf_[0] + self.origin_[0]
            y = src["y"] / self.resolution_ - self.real2simu_tf_[1] + self.origin_[1]
            self.src_pos_.append( (int(x), int(y)) )

        # convert microphone position to cell-scale (in which cell is the sound source)
        self.dst_pos_ = []
        # dst 1 start from index 0
        for dst in dst_pos:
            x = dst["x"] / self.resolution_ - self.real2simu_tf_[0] + self.origin_[0]
            y = dst["y"] / self.resolution_ - self.real2simu_tf_[1] + self.origin_[1]
            self.dst_pos_.append( (int(x), int(y)) )


    def load_whole_dataset(self):
        self.dataset_ = []
        for file in self.voice_file_dirs_:
            data = np.load(file)
            if self.is_normalize_:
                data = normalize(data)

            self.dataset_.append( data )
        print("[INFO] DataLoader.load_whole_dataset: load whole dataset into memory")


def test():
    voice_data_dir = "../../data/active_voice"
    map_data_dir = "../../data/map/bh9f_lab_map.json"
    pos_tf_dir = "../config/bh9f_pos_tf.json"

    data_loader = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir)
    data_loader.save_segmented_map()

    cnt = 0
    for data in data_loader.voice_data_iterator(n_samples=10, seed=0):
        print("frame #%d: %r, src: %r, dst: %r" % (cnt, data["frames"].shape, 
            data["src"], data["dst"]))
        cnt += 1


if __name__ == '__main__':
    test()