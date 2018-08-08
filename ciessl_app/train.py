import argparse

import numpy as np

from voice_engine.signal_process import stft
from voice_engine.utils import view_spectrum
from model.data_loader import DataLoader


def arg_parser():
    """Argument Parser

    Parse arguments from command line, and perform error checking

    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='voice preprocessing')

    parser.add_argument(
        "--voice",
        dest="voice",
        type=str,
        required=True,
        help="Input voice data directory"
    )
    parser.add_argument(
        "--map",
        dest="map",
        type=str,
        required=True,
        help="Input map data directory"
    )
    
    args = parser.parse_args()

    return args


def train_model(voice_data_dir, map_data_dir):
    data_loader = DataLoader(voice_data_dir, map_data_dir)
    
    cnt = 0
    max_n_frames = -1
    min_n_frames = 100000

    for data in data_loader.voice_data_iterator(seed=0, shuffle=False):
        frames = data["frames"]
        f, t, amp, phase = stft(frames[:, 0], data["samplerate"]) 

        max_n_frames = max(max_n_frames, frames.shape[0])
        min_n_frames = min(min_n_frames, frames.shape[0])

        print("frame #%d: %r" % (cnt, frames.shape))
        print("amp: ", amp.shape)
        print("phase: ", phase.shape)

        # log-scale
        # amp = np.log10(amp)
        # view_spectrum(t, f, amp, "STFT Amp dB")
        # view_spectrum(t, f, phase, "STFT phase")

        cnt += 1

    print("max: %d" % max_n_frames)
    print("min: %d" % min_n_frames)


if __name__ == '__main__':
    args = arg_parser()
    
    train_model(voice_data_dir=args.voice, map_data_dir=args.map)