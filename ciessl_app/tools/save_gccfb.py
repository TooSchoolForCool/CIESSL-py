import os
import sys
import argparse
import pickle

import numpy as np
import json

from voice_engine.signal_process import gccfb
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


def save_gccphat2file(voice_data_dir, map_data_dir, pos_tf_dir, out_path):
    n_channels = 16
    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir)

    voice_cnt = 1
    for voice in dl.voice_data_iterator():
        frames = voice["frames"]
        samplerate = voice["samplerate"]
        room_id = voice["src_room_id"]

        gccfb_feature = []
        for i in range(0, n_channels):
            for j in range(i+1, n_channels):
                gccfb_pair = gccfb(frames[:, i], frames[:, j], samplerate, n_mels=40, f_size=20)
                gccfb_feature.append(gccfb_pair)
        gccfb_feature = np.asarray(gccfb_feature)

        json_data = {"gccfb" : gccfb_feature, "room_id" : room_id}

        file_name = "sample-" + str(voice_cnt) + ".pickle"
        with open(out_path + "/" + file_name, "wb") as out_file:
            pickle.dump(json_data, out_file)
            print("Save GCCFB at {}".format(out_path + "/" + file_name))

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