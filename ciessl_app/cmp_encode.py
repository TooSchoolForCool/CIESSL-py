import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from model.data_loader import DataLoader
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
        "--voice_encoder",
        dest="voice_encoder",
        type=str,
        required=True,
        help="Model directory for voice encoder"
    )

    args = parser.parse_args()

    return args


def main():
    args = arg_parser()

    n_frames = 8000
    t1, t2 = 1, 6
    done_t1, done_t2 = False, False

    voice_enc = VoiceVAE()
    voice_enc.load(args.voice_encoder)

    dl = DataLoader(args.voice_data, args.map_data, args.config)

    for voice in dl.voice_data_iterator(seed=0):
        src_idx = voice["src_idx"]

        if src_idx not in [t1, t2]:
            continue

        frames = voice["frames"]
        frames = frames.T  # frames (n_channels, n_samples)
        frames = frames[:, :n_frames].flatten()
        frames = torch.Tensor(frames)
        frames = Variable(frames)
        if torch.cuda.is_available():
            frames = frames.cuda()

        code = voice_enc.encode(frames)

        # convert code to numpy.ndarray (n_feature, )
        if torch.cuda.is_available():
            code = code.data.cpu().numpy()
        else:
            code = code.data.numpy()

        if not done_t1 and src_idx == t1:
            print("src {}".format(t1))
            plt.plot(code, color='b')
            done_t1 = True
        
        if not done_t2 and src_idx == t2:
            print("src {}".format(t2))
            plt.plot(code, color='r')
            done_t2 = True

        if done_t1 and done_t2:
            break

    plt.show()



if __name__ == '__main__':
    main()