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

    n_freq_bins = 255
    n_time_bins = 255

    def min_max_scaler(data):
        # log-scale transform
        data = np.log10(data)
        for i in range(data.shape[0]):
            min_val = np.amin(data[i])
            max_val = np.amax(data[i])
            data[i] = 1.0 * (data[i] - min_val) / (max_val - min_val)
        return data

    def append_func(dataset, data):
        for d in data:
            dataset.append( [d[:n_freq_bins, :n_time_bins]] )
        return dataset

    dl = BatchLoader(args.dataset, scaler=min_max_scaler, mode="all", append_data=append_func)

    for batch in dl.load_batch(batch_size=8, flatten=False):
        data = torch.Tensor(batch)
        data = Variable(data)

        if torch.cuda.is_available():
            data = data.cuda()

        recon_data = encoder.forward(data)
    
        # convert code to numpy.ndarray (n_feature, )
        if torch.cuda.is_available():
            recon_data = recon_data.data.cpu().numpy()
        else:
            recon_data = recon_data.data.numpy()

        for original, recon in zip(batch, recon_data):
            plt.pcolormesh(original[0])
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title("original")
            plt.colorbar()
            plt.show()

            plt.pcolormesh(recon[0])
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title("recon")
            plt.colorbar()
            plt.show()

        exit(0)

        recon_data = map(lambda x:x+shift, recon_data)

        plt.plot(data, color="b", label="original")
        plt.plot(recon_data, color="r", label="reconstructed")
        plt.legend(loc=0)
        plt.show()



if __name__ == '__main__':
    main()