import argparse

import numpy as np

from voice_engine.signal_process import stft
from voice_engine.utils import view_spectrum
from model.data_loader import DataLoader
from model.ranksvm import RankSVM
from model.pipeline import Pipeline
from model.evaluator import Evaluator


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
    rank_svm = RankSVM(max_iter=100, alpha=0.01, loss='squared_loss')
    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir, verbose=False)
    map_data = dl.load_map_info()

    pipe = Pipeline()

    # preparing init training set
    init_training_X = None
    init_training_y = None
    for voice in dl.voice_data_iterator(n_samples=10, seed=0):
        X, y = pipe.prepare_training_data(map_data, voice, n_frames=21000)
        if init_training_X is None:
            init_training_X = X
            init_training_y = y
        else:
            init_training_X = np.append(init_training_X, X, axis=0)
            init_training_y = np.append(init_training_y, y, axis=0)

    rank_svm.fit(X, y)

    cnt = 1
    evaluator = Evaluator(map_data["n_room"])
    for voice in dl.voice_data_iterator(seed=1):
        print("sample %d: src %d: %r" % (cnt, voice["src_idx"], voice["src"]))
        X, y = pipe.prepare_training_data(map_data, voice, n_frames=21000)
        predicted_y = rank_svm.predict(X)

        evaluator.evaluate(y, predicted_y)

        print("sample %d: %r" % (cnt, evaluator.get_eval_result()))
        rank_svm.partial_fit(X, y)
        cnt += 1


if __name__ == '__main__':
    args = arg_parser()
    
    train_model(voice_data_dir=args.voice, map_data_dir=args.map, pos_tf_dir=args.config)