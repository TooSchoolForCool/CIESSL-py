import sys
import os
import json
import Queue

import numpy as np
from voice_engine.signal_process import stft
from voice_engine.utils import view_spectrum

sys.path.append(os.path.abspath(os.path.join("..")))
from utils import save_segmented_map, show_flooding_map

class Pipeline(object):
    """
    Data processing pipeline. Preparing training and testing samples for feeding
    into learning model.
    """

    def __init__(self):
        pass


    def prepare_training_data(self, map_data, voice_data):
        """
        This function is used to prepare training data, which acquiring map and voice 
        data from DataLoader, and return corresponding feature vectors and lebals.

        Args:
            map_data (dictionary):
                map_data["data"] ( np.ndarray (height, width) ): 2D room occupancy grid map,
                    each item is an integer in range(0, N). 0 represents wall, and 1 ~ N 
                    represents room index
                map_data["n_room"] (int): number of rooms
                map_data["origin"] ( tuple (int, int) ): the index of the origin of the map
                map_data["center"] ( list of 2-item tuple (x, y) ): each 2-item tuple
                    represents a center of a room
            voice_data (dictionary):
                voice_data["samplerate"] (int): samplerate of the voice data
                voice_data["src"] (int, int): coordinate of the sound source in the map
                voice_data["dst"] (int, int): coordinate of the microphone in the map
                voice_data["frames"] ( np.ndarray (n_samples, n_channels) ): 
                    sound signal frames from every mic channel
        Returns:

        """
        # save_segmented_map(map_data["data"], map_data["n_room"], 
        #     map_data["center"], map_data["origin"], [voice_data["src"]], 
        #     [voice_data["dst"]], output_path="segmented_map.png")
        src_room = self.__get_room_idx(map_data["data"], voice_data["src"][0], voice_data["dst"][1])
        sound_fading_rate = 0.998
        mic_fading_rate = 0.998

        for i in range(0, map_data["n_room"]):
            src_flooding_map = self.__flooding_map(map_data["data"], map_data["center"][i], 
                sound_fading_rate)
            # show_flooding_map(src_flooding_map)

            mic_flooding_map = self.__flooding_map(map_data["data"], voice_data["dst"],
                mic_fading_rate)
            # show_flooding_map(mic_flooding_map)

            product_map = self.__product_mask(src_flooding_map, mic_flooding_map)
            show_flooding_map(product_map)


        print("src: {}, dst: {}".format(voice_data["src"], voice_data["dst"]))

        amp, phase = self.__stft(voice_data["frames"][:, 0], voice_data["samplerate"])


    def prepare_inference_data(self):
        pass


    def __product_mask(self, ma, mb):
        assert(ma.shape == mb.shape)
        product_map = np.zeros((ma.shape[0], ma.shape[1], 1), np.uint8)

        for y in range(0, ma.shape[0]):
            for x in range(0, ma.shape[1]):
                product_map[y, x] = int((1.0 * ma[y, x] * mb[y, x]) / 255)

        return product_map



    def __flooding_map(self, grid_map, src, rate):
        flooding_map = np.zeros((grid_map.shape[0], grid_map.shape[1], 1), np.uint8)
        visited = np.zeros(grid_map.shape, np.uint8)

        pivot = 255.0
        height, width = grid_map.shape

        q = Queue.Queue()
        q.put( (src[0], src[1]) )
        visited[src[1], src[0]] = 1

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

                # if x + 1 < width and y + 1 < height and visited[y + 1, x + 1] == 0:
                #     q.put( (x + 1, y + 1) )
                #     visited[y + 1, x + 1] = 1
                # if x - 1 >= 0 and y + 1 < height and visited[y + 1, x - 1] == 0:
                #     q.put( (x - 1, y + 1) )
                #     visited[y + 1, x - 1] = 1
                # if y - 1 >= 0 and x + 1 < width and visited[y - 1, x + 1] == 0:
                #     q.put( (x + 1, y - 1) )
                #     visited[y - 1, x + 1] = 1
                # if y - 1 >= 0 and x - 1 >= 0 and visited[y - 1, x - 1] == 0:
                #     q.put( (x - 1, y - 1) )
                #     visited[y - 1, x - 1] = 1

                # print("paint (%d, %d) with %d" % (x, y, int(pivot)))
                flooding_map[y, x] = int(pivot)

            pivot *= rate

        return flooding_map



    def __get_room_idx(self, segmented_map, x, y):
        if segmented_map[y, x] == 0:
            print("[ERROR] (%d, %d) is wall" % (x, y))
            exit(0)
        else:
            return segmented_map[y, x]


    def __stft(self, frames, sample_rate):
        f, t, amp, phase = stft(frames, sample_rate)
        return amp, phase


def test():
    from data_loader import DataLoader

    voice_data_dir = "../../data/active_voice"
    map_data_dir = "../../data/map/bh9f_lab_map.json"
    pos_tf_dir = "../config/bh9f_pos_tf.json"

    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir)
    map_data = dl.load_map_info()

    pipe = Pipeline()
    for voice in dl.voice_data_iterator(n_samples=1):
        ret = pipe.prepare_training_data(map_data, voice)


if __name__ == '__main__':
    test()