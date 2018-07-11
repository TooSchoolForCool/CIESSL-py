import numpy as np
from scipy.io import wavfile

def write2wav(frames, n_channels, rate, output=None):
    frames_per_channel = len(frames) / n_channels
    frames = np.reshape(frames, (frames_per_channel, n_channels))

    #Write multi-channel .wav file with SciPy
    if output is None:
        output = "%dch_%dHz.wav" % (n_channels, rate)

    wavfile.write(output, rate, frames)
    print("[INFO] wave file is stored at: %s" % output)