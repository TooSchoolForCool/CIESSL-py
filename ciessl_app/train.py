import argparse

import numpy as np

from voice_engine.signal_process import stft
from voice_engine.utils import view_spectrum
from model.data_loader import DataLoader
from model.ranksvm import RankSVM
from model.pipeline import Pipeline


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
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True,
        help="Training set configuration file directory"
    )
    
    args = parser.parse_args()

    return args


def train_model(voice_data_dir, map_data_dir, pos_tf_dir):
    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir)
    map_data = dl.load_map_info()

    pipe = Pipeline()

    for voice in dl.voice_data_iterator(n_samples=5):
        X, y = pipe.prepare_training_data(map_data, voice, n_frames=21000)
        print(X.shape)
        print(y.shape)


if __name__ == '__main__':
    args = arg_parser()
    
    train_model(voice_data_dir=args.voice, map_data_dir=args.map, pos_tf_dir=args.config)