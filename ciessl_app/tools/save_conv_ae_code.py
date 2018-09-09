import os
import sys
import argparse
import random

import numpy as np
from sklearn.preprocessing import normalize
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# import DataLoader
sys.path.append(os.path.dirname(__file__) + "/../")
from model.autoencoder import VoiceConvAE
from model.batch_loader import BatchLoader
from utils import load_encoder_model


def arg_parser():
    """Argument Parser

    Parse arguments from command line, and perform error checking

    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='Training Voice AutoEncoder')

    parser.add_argument(
        "--data",
        dest="data",
        type=str,
        required=True,
        help="Input data directory"
    )

    parser.add_argument(
        "--out",
        dest="out",
        type=str,
        required=True,
        help="output directory"
    )

    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        required=True,
        help="encoder model directory"
    )
    
    args = parser.parse_args()

    return args


def save_ae_code(voice_data_dir, out_path, model_path):    
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

    bl = BatchLoader(voice_data_dir, scaler=min_max_scaler, mode="all", append_data=append_func)
    model = load_encoder_model(model_path)

    code_batch = None
    for batch in bl.load_batch(batch_size=8, suffle=True, flatten=False):
        data = torch.Tensor(batch)
        data = Variable(data)
        if torch.cuda.is_available():
            data = data.cuda()
        # forward
        code = model.encode(data)
        # flatten tensor (16, x, y, z) ===> (16, x*y*z)
        code = code.view(code.size(0), -1)
        # convert code to numpy.ndarray (n_feature, )
        if torch.cuda.is_available():
            code = code.data.cpu().numpy()
        else:
            code = code.data.numpy()

        if code_batch is None:
            code_batch = code
        else:
            code_batch = np.append(code_batch, code, axis=0)
        print(code_batch.shape)

    code_batch.dump(out_path + "/conv_code_train.pickle")


def save_ae_flatten_code(voice_data_dir, out_path, model_path):    
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
        batch = []
        for d in data:
            batch.append( [d[:n_freq_bins, :n_time_bins]] )
        batch = np.asarray(batch)
        dataset.append(batch)
        return dataset

    bl = BatchLoader(voice_data_dir, scaler=min_max_scaler, mode="all", append_data=append_func)
    model = load_encoder_model(model_path)

    code_batch = []
    for batch in bl.load_batch(batch_size=1, suffle=False, flatten=False):
        data = torch.Tensor(batch[0])
        data = Variable(data)
        if torch.cuda.is_available():
            data = data.cuda()
        # forward
        code = model.encode(data)
        # flatten tensor (16, x, y, z) ===> (16, x*y*z)
        code = code.view(code.size(0), -1)
        # convert code to numpy.ndarray (n_feature, )
        if torch.cuda.is_available():
            code = code.data.cpu().numpy()
        else:
            code = code.data.numpy()
        code = code.flatten()

        code_batch.append(code)
        print(len(code_batch))

    code_batch = np.asarray(code_batch)
    print(code_batch.shape)
    code_batch.dump(out_path + "/conv_code_train.pickle")


def main():
    args = arg_parser()

    # check output directory if exists
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    save_ae_flatten_code(args.data, args.out, args.model)

if __name__ == '__main__':
    main()