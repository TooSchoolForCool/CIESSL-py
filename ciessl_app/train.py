import argparse

import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from skmultilearn.adapt import MLkNN, MLARAM

from voice_engine.signal_process import stft
from voice_engine.utils import view_spectrum
from model.data_loader import DataLoader
from model.ranksvm import RankSVM
from model.pipeline import Pipeline
from model.evaluator import Evaluator
from model.autoencoder import VoiceVAE, VoiceEncoder
from model.online_clf import OnlineClassifier
from model.rank_clf import RankCLF
from model.rank_fogd import RankFOGD
from model.trace_tracker import TraceTracker
import utils


def arg_parser():
    """Argument Parser

    Parse arguments from command line, and perform error checking

    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='voice preprocessing')

    parser.add_argument(
        "--voice_data",
        dest="voice_data",
        type=str,
        required=True,
        help="Input voice data directory"
    )
    parser.add_argument(
        "--map_data",
        dest="map_data",
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
        "--mode",
        dest="mode",
        type=str,
        required=True,
        help="choose learning mode: [clf, rank]"
    )

    voice_feature = ["gccphat", "stft", "enc", "gcc_enc", "conv_enc"]
    parser.add_argument(
        "--voice_feature",
        dest="voice_feature",
        type=str,
        required=True,
        help="voice feature: {}".format(voice_feature)
    )

    map_feature = ["flooding"]
    parser.add_argument(
        "--map_feature",
        dest="map_feature",
        type=str,
        # required=True,
        help="map feature: {}".format(map_feature)
    )
    parser.add_argument(
        "--voice_encoder",
        dest="voice_encoder",
        type=str,
        help="Model directory for voice encoder"
    )
    parser.add_argument(
        "--save_trace",
        dest="save_trace",
        type=str,
        help="Output directory of robot exploration traces"
    )

    args = parser.parse_args()

    # Validation
    if args.voice_feature == "enc":
        try:
            assert(args.voice_encoder is not None)
        except:
            print("[ERROR] train: Must specify model path when using autoencoder for" 
                + " extracting voice feature")
            raise

    return args


def init_pipeline(voice_feature, map_feature, voice_encoder_path):
    n_frames=6000
    sound_fading_rate=0.996
    mic_fading_rate=0.996
    gccphat_size=25

    voice_enc = None

    if voice_feature in ["enc", "gcc_enc", "conv_enc"]:
        voice_enc = utils.load_encoder_model(cfg_path=voice_encoder_path)

    pipe = Pipeline(
        n_frames=n_frames,
        sound_fading_rate=sound_fading_rate,
        mic_fading_rate=mic_fading_rate,
        voice_feature=voice_feature,
        gccphat_size=gccphat_size,
        map_feature=map_feature,
        voice_encoder=voice_enc
    )

    return pipe


def classification_mode(voice_data_dir, map_data_dir, pos_tf_dir, voice_feature,
    map_feature, voice_encoder_path, save_trace):
    """
    Treat the task as a classification problem.
    """
    clf = MLPClassifier(solver="adam", learning_rate_init=0.0001)
    l2r = OnlineClassifier(clf, q_size=50, shuffle=True)

    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir, verbose=False)
    map_data = dl.load_map_info()

    pipe = init_pipeline(voice_feature, map_feature, voice_encoder_path)

    # preparing init training set
    init_training_X = None
    init_training_y = None
    for voice in dl.voice_data_iterator(n_samples=10, seed=0):
        X, y = pipe.prepare_training_data(map_data, voice)
        if init_training_X is None:
            init_training_X = X
            init_training_y = y
        else:
            init_training_X = np.append(init_training_X, X, axis=0)
            init_training_y = np.append(init_training_y, y, axis=0)

    classes = [i for i in range(1, map_data["n_room"] + 1)]
    print(init_training_X.shape)
    print(init_training_y.shape)
    # l2r.partial_fit(init_training_X, init_training_y, classes=classes, n_iter=10)
    l2r.fit(init_training_X, init_training_y)

    cnt = 1
    evaluator = Evaluator(map_data["n_room"])
    tracker = TraceTracker(verbose=True)
    for voice in dl.voice_data_iterator(seed=7):
        # print("sample %d: src %d: %r" % (cnt, voice["src_idx"], voice["src"]))
        X, y = pipe.prepare_training_data(map_data, voice)

        # predicted_y = l2r.predict_proba(X).todense()
        # predicted_y = np.asarray(predicted_y)
        predicted_y = l2r.predict_proba(X)

        print("Sample %d" % cnt)
        print("y:\t%r" % (y))
        print("pred:\t {}".format(predicted_y))

        evaluator.evaluate(y, predicted_y)
        print("acc: %r" % (evaluator.get_eval_result()))

        if save_trace is not None:
            tracker.append(predicted_y[0], y[0], voice["mic_room_id"])
        # l2r.partial_fit(X, y, n_iter=10)
        l2r.fit(X, y)
        cnt += 1

    if save_trace is not None:
        tracker.dump(save_trace)

    evaluator.plot_acc_history()


def ranking_mode(voice_data_dir, map_data_dir, pos_tf_dir, voice_feature,
    map_feature, voice_encoder_path, save_trace):
    """
    Treat the task as a ranking problem.
    """
    # clf = RankSVM(max_iter=100, alpha=0.01, loss='squared_loss')
    # clf = MLkNN(k=10)
    clf = MLARAM(vigilance=0.9, threshold=0.02)
    # clf = RankCLF(n_classes=4, C=1.0, n_iter=1)
    # clf = RankFOGD(n_classes=4, eta=1e-3, D=10000, sigma=10.0)
    l2r = OnlineClassifier(clf, q_size=50, shuffle=True)

    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir, verbose=False)
    map_data = dl.load_map_info()

    pipe = init_pipeline(voice_feature, map_feature, voice_encoder_path)

    # preparing init training set
    init_training_X = None
    init_training_y = None
    for voice in dl.voice_data_iterator(n_samples=1, seed=0):
        X, y = pipe.prepare_training_data(map_data, voice)
        
        new_y = []
        for i in range(1, map_data["n_room"] + 1):
            new_y.append(1 if i == y else -1)
        y = np.asarray([new_y])

        if init_training_X is None:
            init_training_X = X
            init_training_y = y
        else:
            init_training_X = np.append(init_training_X, X, axis=0)
            init_training_y = np.append(init_training_y, y, axis=0)

    classes = [i for i in range(1, map_data["n_room"] + 1)]
    print(init_training_X.shape)
    print(init_training_y.shape)
    # l2r.partial_fit(init_training_X, init_training_y, classes=classes, n_iter=5)
    l2r.fit(init_training_X, init_training_y)
    # l2r.partial_fit(init_training_X, init_training_y, n_iter=5)

    cnt = 1
    evaluator = Evaluator(map_data["n_room"])
    tracker = TraceTracker(verbose=True)
    for voice in dl.voice_data_iterator(n_samples=5, seed=7):
        # print("sample %d: src %d: %r" % (cnt, voice["src_idx"], voice["src"]))
        X, y = pipe.prepare_training_data(map_data, voice)

        new_y = []
        for i in range(1, map_data["n_room"] + 1):
            new_y.append(1 if i == y else -1)
        new_y = np.asarray([new_y])

        # predicted_y = l2r.predict_proba(X).todense()
        # predicted_y = np.asarray(predicted_y)
        predicted_y = l2r.predict_proba(X)

        print("*" * 60)
        print("Sample %d" % cnt)
        print("y:\t%r" % (y))
        print("pred:\t {}".format(predicted_y[0]))
        evaluator.evaluate(y, predicted_y)
        print("acc: %r" % (evaluator.get_eval_result()))

        if save_trace is not None:
            tracker.append(predicted_y[0], y[0], voice["mic_room_id"])

        # l2r.partial_fit(X, y, n_iter=5)
        l2r.fit(X, new_y)
        cnt += 1

    if save_trace is not None:
        tracker.dump(save_trace)

    evaluator.plot_acc_history()


def train_model():
    args = arg_parser()

    if args.mode == "clf":
        classification_mode(voice_data_dir=args.voice_data, map_data_dir=args.map_data, 
            pos_tf_dir=args.config, voice_feature=args.voice_feature, map_feature=args.map_feature,
            voice_encoder_path=args.voice_encoder, save_trace=args.save_trace)
    elif args.mode == "rank":
        ranking_mode(voice_data_dir=args.voice_data, map_data_dir=args.map_data, 
            pos_tf_dir=args.config, voice_feature=args.voice_feature, map_feature=args.map_feature,
            voice_encoder_path=args.voice_encoder, save_trace=args.save_trace)


if __name__ == '__main__':
    train_model()