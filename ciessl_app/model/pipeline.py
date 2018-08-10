import json

from voice_engine.signal_process import stft
from voice_engine.utils import view_spectrum

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
                voice_data["samplerate"] (int): sample rate of the voice data
                voice_data["src"] (int): source index (where sound source is placed)
                voice_data["dst"] (int): destination index (where mic array is placed)
                voice_data["room"] (int): room index (in which room mic array is placed)
                voice_data["idx"] (int): sample index (the i-th sample that in the <src, dst> dataset)
                voice_data["frames"] ( np.ndarray (n_samples, n_channels) ): 
                    sound signal frames from every mic channel
        Returns:

        """
        print("src: {}, dst: {}, sample #{}".format(voice_data["src"], voice_data["dst"], 
            voice_data["idx"]))

        amp, phase = self.__stft(voice_data["frames"][:, 0], voice_data["samplerate"])


    def prepare_inference_data(self):
        pass


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