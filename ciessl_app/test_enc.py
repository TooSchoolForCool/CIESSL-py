import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from model.batch_loader import BatchLoader
from utils import load_encoder_model


def arg_parser():
    """Argument Parser

    Parse arguments from command line, and perform error checking

    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='voice preprocessing')

    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        required=True,
        help="Input voice data directory"
    )
    parser.add_argument(
        "--encoder_model",
        dest="encoder",
        type=str,
        required=True,
        help="Model directory for autoencoder"
    )

    args = parser.parse_args()

    return args


def main():
    args = arg_parser()

    encoder = load_encoder_model(args.encoder)
    dl = BatchLoader(args.dataset)

    shift = 0.3

    for data in dl.load_batch(batch_size=1, flatten=False):
        data = data.T  # frames (n_channels, n_samples)
        data = data.flatten()
        tensor_data = torch.Tensor(data)
        tensor_data = Variable(tensor_data)
        if torch.cuda.is_available():
            tensor_data = tensor_data.cuda()

        recon_data = encoder.forward(tensor_data)

        # convert code to numpy.ndarray (n_feature, )
        if torch.cuda.is_available():
            recon_data = recon_data.data.cpu().numpy()
        else:
            recon_data = recon_data.data.numpy()

        recon_data = map(lambda x:x+shift, recon_data)

        plt.plot(data, color="b", label="original")
        plt.plot(recon_data, color="r", label="reconstructed")
        plt.legend(loc=0)
        plt.show()



if __name__ == '__main__':
    main()