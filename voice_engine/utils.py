import numpy as np
import samplerate as sr
import matplotlib.pyplot as plt
from scipy.io import wavfile


def write2wav(frames, n_channels, rate, output=None):
    #Write multi-channel .wav file with SciPy
    if output is None:
        output = "%dch_%dHz.wav" % (n_channels, rate)

    wavfile.write(output, rate, frames)
    print("[INFO] wave file is stored at: %s" % output)


def resample(raw_frames, input_rate, target_rate, n_channels, dtype, converter_type='linear'):
    resampler = sr.Resampler(converter_type=converter_type, channels=n_channels)

    ratio = 1.0 * target_rate / input_rate
    resampled_frames = resampler.process(raw_frames, ratio).astype(dtype)

    return resampled_frames


def view_spectrum(time, freq, spec, title, save=False):
    plt.pcolormesh(time, freq, spec)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)

    if save:
        plt.savefig(title + ".png")
    else:
        plt.show()


def view_gccphat(spec_grid, title, save=False):
    plt.pcolormesh(spec_grid)
    plt.title(title)

    plt.ylabel("GCC-PHAT Feature")
    plt.xlabel("Pair Index")
    plt.colorbar()

    if save:
        plt.savefig(title + ".png")
    else:
        plt.show()
