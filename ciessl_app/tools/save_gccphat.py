import os
import sys
import argparse

import numpy as np

from voice_engine.signal_process import gcc_phat
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
    
    args = parser.parse_args()

    return args


def calc_gccphat(frames, samplerate):
    """
    Calculate GCC-PHAT feature

    Args:
        frames (np.ndarray (n_samples, n_channels)): voice frames
        samplerate: voice sample rate

    Returns:
        gccphat_pattern (np.ndarray (gccphat_size, n_pairs)): GCC-PHAT features
    """
    gccphat_size = 25
    n_channels = frames.shape[1]
    gccphat_pattern = []

    for i in range(0, n_channels):
        for j in range(i+1, n_channels):
            tau, cc, center = gcc_phat(frames[:, i], frames[:, j], samplerate)
            # crop gcc_phat features
            if center - gccphat_size < 0:
                cc_feature = cc[0 : 2 * gccphat_size]
            elif center + gccphat_size >= cc.shape[0]:
                cc_feature = cc[-2 * gccphat_size : ]
            else:
                cc_feature = cc[center - gccphat_size : center + gccphat_size]

            # check feature size
            try:
                assert(cc_feature.shape[0] == 2 * gccphat_size)
            except:
                print("[ERROR] __extract_gccphat: gcc_phat feature size does not" + 
                    " match want size %d but what actually get is: %d" % 
                    (2 * gccphat_size, cc_feature.shape[0]))
                print("cc shape: %r" % (cc.shape))
                print("center: %d" % center)
                raise
            
            gccphat_pattern.append(cc_feature)

    # (gccphat_size, n_pairs)
    # gccphat_pattern[:, 0]: pair 0 gccphat feature
    gccphat_pattern = np.asarray(gccphat_pattern).T
    return gccphat_pattern


def save_gccphat2file(voice_data_dir, map_data_dir, pos_tf_dir, out_path):

    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir)

    voice_cnt = 1
    for voice in dl.voice_data_iterator():
        frames = voice["frames"]
        samplerate = voice["samplerate"]

        # gccphat_pattern (np.ndarray (gccphat_size, n_pairs)): GCC-PHAT features
        gccphat_pattern = calc_gccphat(frames, samplerate)

        output_path = out_path + '/' + "gccphat-" + str(voice_cnt) + ".pickle"
        gccphat_pattern.dump(output_path)
        print("[INFO] GCC-PHAT is saved at: %s" % (output_path))

        voice_cnt += 1


def main():
    args = arg_parser()

    # check output directory if exists
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    save_gccphat2file(voice_data_dir=args.voice, map_data_dir=args.map,
        pos_tf_dir=args.config, out_path=args.out)

    
if __name__ == '__main__':
    main()