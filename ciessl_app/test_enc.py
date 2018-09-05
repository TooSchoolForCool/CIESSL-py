import argparse

import numpy as np
import matplotlib.pyplot as plt
import librosa
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
    
    freq = librosa.core.fft_frequencies(sr=48000, n_fft=1024)
    freq_bins = [int(freq[i]) for i in range(0, n_freq_bins)]
    freq_bins[-1] = 12000
    time_bins = [i for i in range(0, n_time_bins)]

    dl = BatchLoader(args.dataset, scaler=min_max_scaler, mode="all", append_data=append_func)

    for batch in dl.load_batch(batch_size=16, flatten=False, suffle=False):
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

        cnt = 1
        for original, recon in zip(batch, recon_data):
            # # spectrum only
            # ax = plt.axes([0,0,1,1], frameon=False)
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            # plt.xlim(0, 255)
            # plt.ylim(0, 255)
            # frames = plt.pcolormesh(original[0])
            # plt.savefig("origin_spec_" + str(cnt) + ".png", transparent=True, dpi=300)
            # plt.clf()

            plt.pcolormesh(time_bins, freq_bins, recon[0])
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.xlim(0, 255)
            # plt.title("Reconstructed Spectrum")
            plt.colorbar()
            # plt.show()
            plt.savefig("origin_" + str(cnt) + ".png", transparent=True, dpi=300)
            plt.clf()

            plt.pcolormesh(time_bins, freq_bins, recon[0])
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.xlim(0, 255)
            # plt.title("Reconstructed Spectrum")
            plt.colorbar()
            # plt.show()
            plt.savefig("recon_" + str(cnt) + ".png", transparent=True, dpi=300)
            plt.clf()

            print(cnt)
            cnt += 1

        exit(0)

        recon_data = map(lambda x:x+shift, recon_data)

        plt.plot(data, color="b", label="original")
        plt.plot(recon_data, color="r", label="reconstructed")
        plt.legend(loc=0)
        plt.show()



if __name__ == '__main__':
    main()