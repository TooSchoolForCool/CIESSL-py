import os
import sys
import argparse

import numpy as np

from voice_engine.signal_process import gcc_phat
from voice_engine.signal_process import stft
from voice_engine.utils import view_spectrum

# import DataLoader
sys.path.append(os.path.dirname(__file__) + "/../")
from model.data_loader import DataLoader


def arg_parser():
    """Argument Parser

    Parse arguments from command line, and perform error checking

    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='Save GCC-PHAT to local files')

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

    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True,
        help="Training set configuration file directory"
    )

    parser.add_argument(
        "--out",
        dest="out",
        type=str,
        required=True,
        help="output directory"
    )

    parser.add_argument(
        "--normalize",
        dest="normalize",
        action='store_true',
        help="Normalize the data or not"
    )
    parser.set_defaults(normalize=False)
    
    args = parser.parse_args()

    return args


def normalize(spec, type):
    """
    Log-scale spectrum first, then scale every points into [0, 1] by adopting
    Min-Max Scaler
    """
    if type == "amp":
        spec = np.log10(spec)

    min_val = np.amin(spec)
    max_val = np.amax(spec)
    spec = 1.0 * (spec - min_val) / (max_val - min_val)

    return spec


def save_stft(voice_data_dir, map_data_dir, pos_tf_dir, out_path, if_normalize=False):
    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir)

    voice_cnt = 1
    for voice in dl.voice_data_iterator():
        frames = voice["frames"]
        samplerate = 48000

        amp_stack = []
        phase_stack = []
        # gccphat_pattern (np.ndarray (gccphat_size, n_pairs)): GCC-PHAT features
        for i in range(0, 16):
            freqs, time, amp, phase = stft(frames[:24000, i], samplerate, nfft=1024, 
                segment_size=256, overlap_size=224)

            if if_normalize:
                amp = normalize(amp, type="amp")
                phase = normalize(phase, type="phase")

            amp_stack.append(amp)
            phase_stack.append(phase)

        amp_output_path = out_path + '/amp/' + "stft-amp-" + str(voice_cnt) + ".pickle"
        phase_output_path = out_path + '/phase/' + "stft-phase-" + str(voice_cnt) + ".pickle"

        # data is saved in form of (n_channels, n_freq_bins, n_time_bins)
        amp_stack = np.asarray(amp_stack)
        phase_stack = np.asarray(phase_stack)

        amp_stack.dump(amp_output_path)
        phase_stack.dump(phase_output_path)

        print("[INFO] amp file is saved at: %s" % (amp_output_path))
        print("[INFO] phase file is saved at: %s" % (phase_output_path))

        voice_cnt += 1


def main():
    args = arg_parser()

    # check output directory if exists
    if not os.path.exists(args.out + "/amp") or not os.path.exists(args.out + "/phase"):
        os.makedirs(args.out + "/amp")
        os.makedirs(args.out + "/phase")

    save_stft(voice_data_dir=args.voice, map_data_dir=args.map,
        pos_tf_dir=args.config, out_path=args.out, if_normalize=args.normalize)

    
if __name__ == '__main__':
    main()