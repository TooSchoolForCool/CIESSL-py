import numpy as np
import matplotlib.pyplot as plt

from voice_engine.signal_process import stft
from model.data_loader import DataLoader


def train_model():
    voice_data_dir = "../data/active_voice"
    map_data_dir = "../data/map"

    data_loader = DataLoader(voice_data_dir, map_data_dir)
    
    cnt = 0
    for data in data_loader.voice_data_iterator(n_samples=5, seed=0):
        frames = data["frames"]
        f, t, amp, phase = stft(frames[:, 0], data["samplerate"])
        
        print("frame #%d: %r" % (cnt, frames.shape))
        print("amp: ", amp.shape)
        print("phase: ", phase.shape)

        # log-scale
        amp = np.log10(amp)
        plt.pcolormesh(t, f, amp)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title("STFT dB")
        plt.show()

        plt.pcolormesh(t, f, phase)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title("STFT Phase")
        plt.show()
        cnt += 1

if __name__ == '__main__':
    train_model()