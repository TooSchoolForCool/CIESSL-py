import sys
import os
import json
import Queue

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

from voice_engine.signal_process import stft, gcc_phat
from voice_engine.utils import view_spectrum, view_gccphat

sys.path.append(os.path.abspath(os.path.join("..")))
from utils import save_segmented_map, show_flooding_map


class Pipeline(object):
    """
    Data processing pipeline. Preparing training and testing samples for feeding
    into learning model.
    """
    def __init__(self, n_frames=18000, sound_fading_rate=0.998, mic_fading_rate=0.998,
        gccphat_size=15, voice_feature="gccphat", map_feature=None, voice_encoder=None):
        """
        Constructor

        Args:
            n_frames (int): number of frames used to extract acoustic feature
            sound_fading_rate (double): fading rate of sound flooding map
            mic_fading_rate (double): fading rate of mic flooding map
            gccphat_size (int): the size of gcc_phat pattern (2 * gccphat_size + 1).
                We extract the cross-correlation around the center of gcc_phat
        """
        self.n_frames_ = n_frames
        self.sound_fading_rate_ = sound_fading_rate
        self.mic_fading_rate_ = mic_fading_rate
        self.gccphat_size_ = gccphat_size
        self.voice_feature_ = voice_feature
        self.map_feature_ = map_feature

        self.voice_encoder_ = voice_encoder


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
                map_data["boundary"] (dict): boundary information of the map, bottom-left corner,
                    and the top-right corner
            voice_data (dictionary):
                voice_data["samplerate"] (int): samplerate of the voice data
                voice_data["src"] (int, int): coordinate of the sound source in the map
                voice_data["src_idx"] (int): sound source index
                voice_data["dst"] (int, int): coordinate of the microphone in the map
                voice_data["frames"] ( np.ndarray (n_samples, n_channels) ): 
                    sound signal frames from every mic channel
        Returns:
            X ( np.array (n_samples, n_features) ): feature set
            y ( np.array (n_samples,) ): label set
        """
        X = []
        y = []

        src_room = self.__get_room_idx(map_data["data"], voice_data["src"][0], voice_data["src"][1])

        ######################################################
        # Extract sound feature
        ######################################################
        frame_stack = voice_data["frames"]
        samplerate = voice_data["samplerate"]
        sound_feature = None
        
        if self.voice_feature_ == "stft":
            sound_feature = self.__extract_stft(frame_stack, samplerate)
        elif self.voice_feature_ == "gccphat":
            sound_feature = self.__extract_gccphat(frame_stack, samplerate).flatten()
        elif self.voice_feature_ == "enc":
            sound_feature = self.__encode_voice(frame_stack)
        elif self.voice_feature_ == "gcc_enc":
            sound_feature = self.__encode_gcc(frame_stack, samplerate)
        elif self.voice_feature_ == "conv_enc":
            sound_feature = self.__conv_encode_voice(frame_stack, samplerate)

        ######################################################
        # Extract Map feature
        ######################################################
        map_feature = None

        if self.map_feature_ == "flooding":
            mic_flooding_map = self.__flooding_map(map_data["data"], voice_data["dst"],
                map_data["boundary"], self.mic_fading_rate_)
            shrink_map = self.__shrink_map(mic_flooding_map, kernel_size=(5, 5))
            map_feature = shrink_map.flatten()

        feature_vec = np.append(sound_feature, map_feature)

        # X.append(sound_feature)
        X.append(feature_vec)
        y.append(src_room)

        X = np.asarray(X)
        y = np.asarray(y)

        return X, y


    def __encode_voice(self, frame_stack):
        """
        Calculate gccphat pattern

        Args:
            frame_stack (np.ndarray (n_samples, n_chennals)): Audio frame stack
            samplerate (int): audio source sample rate

        Returns:
            voice_code (np.ndarray (encode_features, )): encoded features
        """
        self.__check_autoencoder()

        # encode voice data
        voice_code = []

        frames = frame_stack.T  # frames (n_channels, n_samples)
        frames = frames[:, 2000:self.n_frames_].flatten()
        frames = torch.Tensor(frames)
        frames = Variable(frames)
        if torch.cuda.is_available():
            frames = frames.cuda()

        code = self.voice_encoder_.encode(frames)

        # convert code to numpy.ndarray (n_feature, )
        if torch.cuda.is_available():
            code = code.data.cpu().numpy()
        else:
            code = code.data.numpy()

        voice_code = code
        # voice_code.append(code)
        # # convert voice_code to numpy.ndarray (n_features, n_channels)
        # voice_code = np.asarray(voice_code).T

        return voice_code

    
    def __check_autoencoder(self):
        try:
            assert(self.voice_encoder_ is not None)
        except:
            print("[ERROR] Pipeline.__check_autoencoder(): must initialize voice_encoder first")
            raise


    def __encode_gcc(self, frame_stack, samplerate):
        self.__check_autoencoder()

        gccphat_pattern = self.__extract_gccphat(frame_stack, samplerate)
        gccphat_pattern = gccphat_pattern.T.flatten()

        gccphat_pattern = torch.Tensor(gccphat_pattern)
        gccphat_pattern = Variable(gccphat_pattern)
        if torch.cuda.is_available():
            gccphat_pattern = gccphat_pattern.cuda()

        code = self.voice_encoder_.encode(gccphat_pattern)

        # convert code to numpy.ndarray (n_feature, )
        if torch.cuda.is_available():
            code = code.data.cpu().numpy()
        else:
            code = code.data.numpy()

        voice_code = code
        # voice_code.append(code)
        # # convert voice_code to numpy.ndarray (n_features, n_channels)
        # voice_code = np.asarray(voice_code).T

        return voice_code


    def __conv_encode_voice(self, frame_stack, samplerate):
        # calculate log-scale normalized stft
        voice_stft = []
        for i in range(0, 16):
            _, _, amp, phase = stft(frame_stack[:24000, i], samplerate, nfft=1024, segment_size=256, 
                overlap_size=224)
            cropped = amp[:255, :255]
            log_cropped = np.log10(cropped)
            log_normalized_cropped = self.__min_max_scaler(log_cropped)
            # each channel stft is a (1, 255, 255) tensor
            voice_stft.append( [log_normalized_cropped] )
        # convert to batch-form
        voice_stft = np.asarray( voice_stft )

        voice_stft = torch.Tensor(voice_stft)
        voice_stft = Variable(voice_stft)
        if torch.cuda.is_available():
            voice_stft = voice_stft.cuda()

        code = self.voice_encoder_.encode(voice_stft)
        # flatten tensor (16, x, y, z) ===> (16, x*y*z)
        code = code.view(code.size(0), -1)

        # convert code to numpy.ndarray (n_feature, )
        if torch.cuda.is_available():
            code = code.data.cpu().numpy()
        else:
            code = code.data.numpy()

        return code.flatten()


    def __min_max_scaler(self, data):
        min_val = np.amin(data)
        max_val = np.amax(data)
        data = 1.0 * (data - min_val) / (max_val - min_val)
        return data


    def __product_mask(self, ma, mb, normalize=False):
        assert(ma.shape == mb.shape)
        product_map = np.zeros((ma.shape[0], ma.shape[1], 1), np.uint8)

        for y in range(0, ma.shape[0]):
            for x in range(0, ma.shape[1]):
                product_map[y, x] = int((1.0 * ma[y, x] * mb[y, x]) / 255)

        return product_map


    def __flooding_map(self, grid_map, center, boundary, rate):
        flooding_map = np.zeros((grid_map.shape[0], grid_map.shape[1], 1), np.uint8)
        visited = np.zeros(grid_map.shape, np.uint8)

        pivot = 255.0
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

                flooding_map[y, x] = int(pivot)

            pivot *= rate

        # crop out map region
        min_x, min_y = boundary["bl"][0], boundary["bl"][1]
        max_x, max_y = boundary["tr"][0], boundary["tr"][1]
        flooding_map = flooding_map[min_y:max_y, min_x:max_x]

        return flooding_map


    def __get_room_idx(self, segmented_map, x, y):
        if segmented_map[y, x] == 0:
            print("[ERROR] (%d, %d) is wall" % (x, y))
            exit(0)
        else:
            return segmented_map[y, x]


    def __fixed_len_stft(self, frames, sample_rate, n_frames):
        assert(len(frames) >= n_frames)

        f, t, amp, phase = stft(frames[:n_frames], sample_rate)
        return f, t, amp, phase


    def __extract_stft(self, frame_stack, samplerate):
        flatten_sound_feature = []
        
        for i in range(0, frame_stack.shape[1]):
            f, t, amp, phase = self.__fixed_len_stft(frame_stack[:, i], samplerate, self.n_frames_)
            flatten_sound_feature = np.append(flatten_sound_feature, phase.flatten())
        
        return flatten_sound_feature


    def __extract_gccphat(self, frame_stack, samplerate):
        """
        Calculate gccphat pattern

        Args:
            frame_stack (np.ndarray (n_samples, n_chennals)): Audio frame stack
            samplerate (int): audio source sample rate

        Returns:
            gccphat_pattern (np.ndarray (gccphat_size, n_pairs)): gcc_phat pattern feature
        """
        n_channels = frame_stack.shape[1]

        gccphat_pattern = []
        for i in range(0, n_channels):
            for j in range(i+1, n_channels):
                tau, cc, center = gcc_phat(frame_stack[:, i], frame_stack[:, j], samplerate)
                # crop gcc_phat features
                if center - self.gccphat_size_ < 0:
                    cc_feature = cc[0 : 2 * self.gccphat_size_]
                elif center + self.gccphat_size_ >= cc.shape[0]:
                    cc_feature = cc[-2 * self.gccphat_size_ : ]
                else:
                    cc_feature = cc[center - self.gccphat_size_ : center + self.gccphat_size_]

                # check feature size
                try:
                    assert(cc_feature.shape[0] == 2 * self.gccphat_size_)
                except:
                    print("[ERROR] __extract_gccphat: gcc_phat feature size does not" + 
                        " match want size %d but what actually get is: %d" % 
                        (2 * self.gccphat_size_, cc_feature.shape[0]))
                    print("cc shape: %r" % (cc.shape))
                    print("center: %d" % center)
                    raise
                
                gccphat_pattern.append(cc_feature)

        # (gccphat_size, n_pairs)
        # gccphat_pattern[:, 0]: pair 0 gccphat feature
        gccphat_pattern = np.asarray(gccphat_pattern).T
        return gccphat_pattern


    def __shrink_map(self, img, kernel_size=(3, 3)):
        # reshape image in terms of pytorch conv requirements
        img = img.astype(float)

        # for ir, fr in zip(img, float_img):
        #     for ip, fp in zip(ir, fr):
        #         print(ip, fp)
        #     exit(0)

        img = torch.from_numpy(img.reshape((1, 1, img.shape[0], img.shape[1])))
        img = F.max_pool2d(Variable(img), kernel_size=kernel_size)
        img = img.data.squeeze().numpy().astype("uint8")
        return img


def test():
    from data_loader import DataLoader

    voice_data_dir = "../../data/active_voice"
    map_data_dir = "../../data/map/bh9f_lab_map.json"
    pos_tf_dir = "../config/bh9f_pos_tf.json"

    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir)
    map_data = dl.load_map_info()

    pipe = Pipeline()
    for voice in dl.voice_data_iterator(n_samples=5, seed=1):
        X, y = pipe.prepare_training_data(map_data, voice, voice_feature="gccphat")
        print(X.shape)
        print(y.shape)


if __name__ == '__main__':
    test()