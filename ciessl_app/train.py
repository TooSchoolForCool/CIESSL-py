import argparse
import random
import os
import sys

import numpy as np
from sklearn.linear_model import SGDClassifier
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
from model.dqn import DQN
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

    voice_feature = ["gccphat", "gccfb", "stft", "enc", "gcc_enc", "conv_enc"]
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
    parser.add_argument(
        "--save_train_hist",
        dest="save_train_hist",
        type=str,
        help="Output directory of training history"
    )
    parser.add_argument(
        "--n_trails",
        dest="n_trails",
        type=int,
        default=1,
        help="Number of trails for experiments"
    )
    parser.add_argument(
        "--n_mic",
        dest="n_mic",
        type=int,
        default=16,
        help="Number of microphoen used"
    )
    parser.add_argument(
        "--model_type",
        dest="model_type",
        type=str,
        required=True,
        help="Define learning model type"
    )
    parser.add_argument(
        "--lm_param",
        dest="lm_param",
        type=float,
        default=None,
        help="Learning Model Parameter"
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

    if args.n_mic not in [16, 8, 4, 2]:
        print("[ERROR] does not support {}-mic configuration, n_mic should be 16, 8 or 4".format(args.n_mic))
        raise

    return args


def init_pipeline(voice_feature, map_feature, voice_encoder_path):
    n_frames=6000
    sound_fading_rate=0.996
    mic_fading_rate=0.996
    gccphat_size=25

    voice_enc = None

    if voice_feature in ["enc", "gcc_enc", "conv_enc", "denoise_enc"]:
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


def create_model(model_type, n_rooms, lm_param=None):
    if model_type == "mlp":
        model = MLPClassifier(solver="adam", learning_rate_init=0.0001)
    elif model_type == "svm":
        model = SGDClassifier(loss="modified_huber", alpha=0.001, learning_rate="optimal")
    elif model_type == "haram":
        vigilance = lm_param
        print("[INFO] HARAM set vigilance to {}".format(vigilance))
        model = MLARAM(vigilance=vigilance, threshold=0.02)
    elif model_type == "rank_fogd":
        model = RankFOGD(n_classes=n_rooms, eta=1e-3, D=5000, sigma=8.0)
    elif model_type == "rank_clf":
        model = RankCLF(n_classes=n_rooms, C=1.0, n_iter=1)
    elif model_type == "mlknn":
        model = MLkNN(k=10)
    else:
        print("[ERROR] Do not support type {}".format(type))
        raise

    return model


def update_model(model, model_type, X, y, n_rooms=None):
    if model_type in ["haram", "rank_clf", "mlknn"]:
        model.fit(X, y)
    elif model_type in ["mlp", "rank_fogd", "svm"]:
        if n_rooms is not None:
            classes = [i for i in range(1, n_rooms + 1)]
            model.partial_fit(X, y, classes=classes, n_iter=1)
        else:
            model.partial_fit(X, y, n_iter=1)


def classification_mode(voice_data_dir, map_data_dir, pos_tf_dir, voice_feature,
    map_feature, voice_encoder_path, save_trace, eval_out_dir, n_trails, n_mic, model_type, lm_param):
    """
    Treat the task as a classification problem.
    """
    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir, n_mic, verbose=False)
    map_data = dl.load_map_info()
    n_rooms = map_data["n_room"] - 1
    
    pipe = init_pipeline(voice_feature, map_feature, voice_encoder_path)

    for t in range(0, n_trails):
        clf = create_model(model_type, n_rooms, lm_param)
        l2r = OnlineClassifier(clf, q_size=50, shuffle=True)

        # preparing init training set
        init_X, init_y = utils.init_training_set(dl, pipe, n_samples=1, seed=random.randint(0, 1000), type="clf")
        print(init_X.shape)
        print(init_y.shape)

        update_model(l2r, model_type, init_X, init_y, n_rooms)

        evaluator = Evaluator(n_rooms, verbose=True)
        tracker = TraceTracker(verbose=True)
        for voice in dl.voice_data_iterator(seed=random.randint(0, 1000)):
            # print("sample %d: src %d: %r" % (cnt, voice["src_idx"], voice["src"]))
            X, y = pipe.prepare_training_data(map_data, voice)

            predicted_y = l2r.predict_proba(X)
            evaluator.evaluate(y, predicted_y)

            if save_trace is not None:
                tracker.append(predicted_y[0], y[0], voice["mic_room_id"])
                
            update_model(l2r, model_type, init_X, init_y)

        if save_trace is not None:
            tracker.dump(save_trace, str(t) + "_trace.json")
        if eval_out_dir is not None:
            evaluator.save_history(out_dir=eval_out_dir, file_prefix=str(t), type="csv")

        # if t == n_trails - 1:
        #     evaluator.plot_acc_history()
        #     evaluator.plot_error_bar(n_bins=20)
    

def ranking_mode(voice_data_dir, map_data_dir, pos_tf_dir, voice_feature,
    map_feature, voice_encoder_path, save_trace, eval_out_dir, n_trails, n_mic, model_type, lm_param):
    """
    Treat the task as a ranking problem.
    """
    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir, n_mic, verbose=False)
    map_data = dl.load_map_info()
    n_rooms = map_data["n_room"] - 1

    pipe = init_pipeline(voice_feature, map_feature, voice_encoder_path)

    for t in range(0, n_trails):
        clf = create_model(model_type, n_rooms, lm_param)
        l2r = OnlineClassifier(clf, q_size=50, shuffle=True)

        # preparing init training set
        init_X, init_y = utils.init_training_set(dl, pipe, n_samples=1, seed=random.randint(0, 1000), type="rank", n_labels=n_rooms)
        print(init_X.shape)
        print(init_y.shape)

        update_model(l2r, model_type, init_X, init_y)

        evaluator = Evaluator(n_rooms, verbose=True)
        tracker = TraceTracker(verbose=True)
        for voice in dl.voice_data_iterator(seed=random.randint(0, 1000)):
            # print("sample %d: src %d: %r" % (cnt, voice["src_idx"], voice["src"]))
            X, y = pipe.prepare_training_data(map_data, voice)
            rank_y = utils.label2rank(label=y, n_labels=n_rooms)
            
            predicted_y = l2r.predict_proba(X)
            evaluator.evaluate(y, predicted_y)

            if save_trace is not None:
                tracker.append(predicted_y[0], y[0], voice["mic_room_id"])

            update_model(l2r, model_type, X, rank_y)

        if save_trace is not None:
            tracker.dump(save_trace, str(t) + "_trace.json")
        if eval_out_dir is not None:
            evaluator.save_history(out_dir=eval_out_dir, file_prefix=str(t), type="csv")

        # if t == n_trails - 1:
        #     evaluator.plot_acc_history()
        #     evaluator.plot_error_bar(n_bins=10)


def rl_mode(voice_data_dir, map_data_dir, pos_tf_dir, voice_feature, map_feature, 
    voice_encoder_path, save_trace, eval_out_dir, n_trails, n_mic, model_type, lm_param):
    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir, n_mic, verbose=False)
    map_data = dl.load_map_info()
    n_rooms = map_data["n_room"] - 1

    pipe = init_pipeline(voice_feature, map_feature, voice_encoder_path)
    X, _ = utils.init_training_set(dl, pipe, n_samples=1, seed=random.randint(0, 1000), type="clf")
    n_states = X.shape[1]

    for t in range(0, n_trails):
        dqn_trainer = DQN(n_states=n_states, n_actions=n_rooms)
        evaluator = Evaluator(n_rooms, verbose=True)
        tracker = TraceTracker(verbose=True)

        for voice in dl.voice_data_iterator(seed=random.randint(0, 1000)):
            X, y = pipe.prepare_training_data(map_data, voice)

            rank_y = dqn_trainer.action_ranking(X[0])
            target_y = np.argmax(rank_y)

            if target_y == y:
                next_s = 1
                reward = -1
            else:
                next_s = 0
                reward = -1
            
            evaluator.evaluate(y, predicted_y)

            if save_trace is not None:
                tracker.append(predicted_y[0], y[0], voice["mic_room_id"])

        if save_trace is not None:
            tracker.dump(save_trace, str(t) + "_trace.json")
        if eval_out_dir is not None:
            evaluator.save_history(out_dir=eval_out_dir, file_prefix=str(t), type="csv")


def train_model():
    args = arg_parser()

    if args.mode == "clf":
        classification_mode(voice_data_dir=args.voice_data, map_data_dir=args.map_data, 
            pos_tf_dir=args.config, voice_feature=args.voice_feature, map_feature=args.map_feature,
            voice_encoder_path=args.voice_encoder, save_trace=args.save_trace, 
            eval_out_dir=args.save_train_hist, n_trails=args.n_trails, n_mic=args.n_mic,
            model_type=args.model_type, lm_param=args.lm_param)
    elif args.mode == "rank":
        ranking_mode(voice_data_dir=args.voice_data, map_data_dir=args.map_data, 
            pos_tf_dir=args.config, voice_feature=args.voice_feature, map_feature=args.map_feature,
            voice_encoder_path=args.voice_encoder, save_trace=args.save_trace,
            eval_out_dir=args.save_train_hist, n_trails=args.n_trails, n_mic=args.n_mic,
            model_type=args.model_type, lm_param=args.lm_param)
    elif args.mode == "rl":
        rl_mode(voice_data_dir=args.voice_data, map_data_dir=args.map_data, 
            pos_tf_dir=args.config, voice_feature=args.voice_feature, map_feature=args.map_feature,
            voice_encoder_path=args.voice_encoder, save_trace=args.save_trace,
            eval_out_dir=args.save_train_hist, n_trails=args.n_trails, n_mic=args.n_mic,
            model_type=args.model_type, lm_param=args.lm_param)


if __name__ == '__main__':
    train_model()