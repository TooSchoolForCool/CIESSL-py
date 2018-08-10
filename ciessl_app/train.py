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
    rank_svm = RankSVM(max_iter=100, alpha=0.01, loss='hinge')
    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir, verbose=True)
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

    print(init_training_X.shape)
    print(init_training_y.shape)

    rank_svm.fit(X, y)

    cnt = 1
    TP, TN, FP, FN = 0, 0, 0, 0
    for voice in dl.voice_data_iterator(seed=1):
        print("sample %d: src %d: %r" % (cnt, voice["src_idx"], voice["src"]))
        X, y = pipe.prepare_training_data(map_data, voice, n_frames=21000)
        predicted_y = rank_svm.predict(X)

        for ty, py in zip(y, predicted_y):
            if ty == py:
                if ty == 1:
                    TP += 1
                else:
                    TN += 0
            else:
                if py == 1:
                    FP += 1
                else:
                    FN += 1

        print("ground truth: %r" % y)
        print("predicted: %r" % predicted_y)

        calc_precision = lambda tp, fp : ((1.0 * tp) / (tp + fp))
        calc_recall = lambda tp, fn : ((1.0 * tp) / (tp + fn))

        print("precision: %.2lf, recall: %.2lf" % (calc_precision(TP, FP), calc_recall(TP, FN)))
        print("-" * 80)

        rank_svm.partial_fit(X, y)
        cnt += 1





if __name__ == '__main__':
    args = arg_parser()
    
    train_model(voice_data_dir=args.voice, map_data_dir=args.map, pos_tf_dir=args.config)