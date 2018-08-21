import argparse

import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier

from voice_engine.signal_process import stft
from voice_engine.utils import view_spectrum
from model.data_loader import DataLoader
from model.ranksvm import RankSVM
from model.pipeline import Pipeline
from model.evaluator import Evaluator
from model.autoencoder import VoiceVAE, VoiceEncoder


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
        help="choose learning mode: [clf, reg]"
    )
    parser.add_argument(
        "--voice_feature",
        dest="voice_feature",
        type=str,
        required=True,
        help="voice feature: [gccphat, stft, enc]"
    )
    parser.add_argument(
        "--map_feature",
        dest="map_feature",
        type=str,
        # required=True,
        help="map feature"
    )
    parser.add_argument(
        "--voice_encoder",
        dest="voice_encoder",
        type=str,
        help="Model directory for voice encoder"
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
    sound_fading_rate=0.999
    mic_fading_rate=0.993
    gccphat_size=15

    voice_enc = None

    if voice_feature == "enc":
        voice_enc = VoiceVAE()
        voice_enc.load(voice_encoder_path)

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
    map_feature, voice_encoder_path):
    """
    Treat the task as a classification problem.
    """

    # rank_svm = RankSVM(max_iter=100, alpha=0.01, loss='squared_loss')
    rank_svm = MLPClassifier(solver="adam")

    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir, verbose=False)
    map_data = dl.load_map_info()

    pipe = init_pipeline(voice_feature, map_feature, voice_encoder_path)

    # preparing init training set
    init_training_X = None
    init_training_y = None
    for voice in dl.voice_data_iterator(n_samples=5, seed=0):
        X, y = pipe.prepare_training_data(map_data, voice)
        if init_training_X is None:
            init_training_X = X
            init_training_y = y
        else:
            init_training_X = np.append(init_training_X, X, axis=0)
            init_training_y = np.append(init_training_y, y, axis=0)

    classes = [i for i in range(1, map_data["n_room"] + 1)]
    rank_svm.partial_fit(init_training_X, init_training_y, classes=classes)

    cnt = 1
    evaluator = Evaluator(map_data["n_room"])
    for voice in dl.voice_data_iterator(seed=7):
        # print("sample %d: src %d: %r" % (cnt, voice["src_idx"], voice["src"]))
        X, y = pipe.prepare_training_data(map_data, voice) 
        predicted_y = rank_svm.predict_proba(X)
        
        evaluator.evaluate(y, predicted_y)

        print("Sample %d" % cnt)
        print("y:\t%r" % (y))
        print("pred:\t%r" % (predicted_y))
        print("acc: %r" % (evaluator.get_eval_result()))

        rank_svm.partial_fit(X, y)
        cnt += 1

    evaluator.plot_acc_history()


def train_model():
    args = arg_parser()

    if args.mode == "clf":
        classification_mode(voice_data_dir=args.voice_data, map_data_dir=args.map_data, 
            pos_tf_dir=args.config, voice_feature=args.voice_feature, map_feature=args.map_feature,
            voice_encoder_path=args.voice_encoder)
    elif args.mode == "reg":
        pass


if __name__ == '__main__':
    train_model()