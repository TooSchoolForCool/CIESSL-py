import os
import argparse
import random

import numpy as np
from sklearn.preprocessing import normalize
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model.autoencoder import VoiceVAE, VoiceEncoder, VoiceConvAE
from model.data_loader import DataLoader
from model.batch_loader import BatchLoader


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

    valid_encoder = ["voice_vae", "voice_ae", "voice_cae", "denoise_ae"]
    parser.add_argument(
        "--encoder",
        dest="encoder",
        type=str,
        required=True,
        help="choose what type of encoder to train: %r" % valid_encoder
    )
    
    args = parser.parse_args()

    # check validation
    if args.encoder not in valid_encoder:
        print("[ERROR] train_autoencoder: --encoder should in {}".format(valid_encoder))
        raise

    return args


def train_voice_vae(voice_data_dir, out_path):
    bl = BatchLoader(voice_data_dir)

    num_epochs = 50000
    batch_size = 8
    learning_rate = 1e-4
    lr_decay_freq = 10
    save_frequency = 100

    model = VoiceVAE()
    if torch.cuda.is_available():
        print("[INFO] CUDA is available")
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    min_loss = 99999999.0

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0

        if epoch % lr_decay_freq == 0 and epoch != 0:
            learning_rate *= 0.99
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

        for batch in bl.load_batch(batch_size=batch_size, suffle=True, flatten=True):
            data = torch.Tensor(batch)
            data = Variable(data)
            if torch.cuda.is_available():
                data = data.cuda()
            # forward
            recon_batch, mu, logvar = model(data)
            loss = model.loss(recon_batch, data, mu, logvar)
            # backward
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('====> Epoch: {}\tAverage loss: {:.6f}\tlearning_rate: {:.7f}'.format(
            epoch, train_loss / bl.size(), learning_rate))

        if epoch != 0 and epoch % save_frequency == 0 and min_loss > train_loss / bl.size():
            min_loss = train_loss / bl.size()
            print("***** model saved at: {}".format(out_path))
            model.save(out_path)


def train_voice_ae(voice_data_dir, out_path):
    bl = BatchLoader(voice_data_dir)

    num_epochs = 50000
    batch_size = 8
    learning_rate = 1e-4
    lr_decay_freq = 10
    save_frequency = 100

    model = VoiceEncoder()
    if torch.cuda.is_available():
        print("[INFO] CUDA is available")
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    min_loss = 99999999.0

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0

        if epoch % lr_decay_freq == 0 and epoch != 0:
            learning_rate *= 0.99
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

        for batch in bl.load_batch(batch_size=batch_size, suffle=True, flatten=True):
            data = torch.Tensor(batch)
            data = Variable(data)
            if torch.cuda.is_available():
                data = data.cuda()
            # forward
            recon_batch = model(data)
            loss = model.loss(recon_batch, data)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print('====> Epoch: {}\tAverage loss: {:.6f}\tlearning_rate: {:.7f}'.format(
            epoch, train_loss / bl.size(), learning_rate))

        if epoch % save_frequency == 0 and min_loss > train_loss / bl.size():
            min_loss = train_loss / bl.size()
            print("***** model saved at: {}".format(out_path))
            model.save(out_path)


def train_voice_cae(voice_data_dir, out_path):
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

    bl = BatchLoader(voice_data_dir, scaler=min_max_scaler, mode="train", append_data=append_func)

    num_epochs = 500000
    batch_size = 4
    learning_rate = 1e-4
    lr_decay_freq = 1000
    save_frequency = 100

    model = VoiceConvAE()
    if torch.cuda.is_available():
        print("[INFO] CUDA is available")
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    min_loss = 99999999.0

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0

        if epoch % lr_decay_freq == 0 and epoch != 0:
            learning_rate *= 0.99
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        
        for batch in bl.load_batch(batch_size=batch_size, suffle=True, flatten=False):
            data = torch.Tensor(batch)
            data = Variable(data)
            if torch.cuda.is_available():
                data = data.cuda()
            # forward
            recon_batch = model(data)
            loss = model.loss(recon_batch, data)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print('====> Epoch: {}\tAverage loss: {:.6f}\tlearning_rate: {:.7f}'.format(
            epoch, train_loss / bl.size(), learning_rate))

        if epoch != 0 and epoch % save_frequency == 0 and min_loss > train_loss / bl.size():
            min_loss = train_loss / bl.size()
            print("***** model saved at: {}".format(out_path))
            model.save(out_path)


def train_denoise_ae(voice_data_dir, out_path):
    # def min_max_scaler(data):
    #     # log-scale transform
    #     for i in range(data.shape[0]):
    #         min_val = np.amin(data[i])
    #         max_val = np.amax(data[i])
    #         data[i] = 1.0 * (data[i] - min_val) / (max_val - min_val)

    #     return data

    def append_func(dataset, data):
        for d in data:
            dataset.append(d)
        return dataset

    # bl = BatchLoader(voice_data_dir, scaler=min_max_scaler, mode="all", append_data=append_func)
    bl = BatchLoader(voice_data_dir, mode="train", append_data=append_func)

    num_epochs = 500000
    batch_size = 8
    learning_rate = 1e-4
    lr_decay_freq = 1000
    save_frequency = 100

    model = VoiceEncoder(nn_structure=[256, 32])
    if torch.cuda.is_available():
        print("[INFO] CUDA is available")
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    min_loss = 99999999.0

    model.train()
    print("dataset size: {}".format(bl.size()))
    for epoch in range(num_epochs):
        train_loss = 0.0

        if epoch % lr_decay_freq == 0 and epoch != 0:
            learning_rate *= 0.99
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        
        for batch in bl.load_batch(batch_size=batch_size, suffle=True, flatten=False):
            data = torch.Tensor(batch)
            data = Variable(data)
            if torch.cuda.is_available():
                data = data.cuda()
            # activation
            data = F.sigmoid(data)
            # forward
            recon_batch = model(data)
            loss = model.loss(recon_batch, data)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print('====> Epoch: {}\tAverage loss: {:.6f}\tlearning_rate: {:.7f}'.format(
            epoch, train_loss / bl.size(), learning_rate))

        if epoch % save_frequency == 0 and min_loss > train_loss / bl.size():
            min_loss = train_loss / bl.size()
            print("***** model saved at: {}".format(out_path))
            model.save(out_path)


def main():
    args = arg_parser()

    if args.encoder == "voice_vae":
        train_voice_vae(voice_data_dir=args.data, out_path=args.out)
    elif args.encoder == "voice_ae":
        train_voice_ae(voice_data_dir=args.data, out_path=args.out)
    elif args.encoder == "voice_cae":
        train_voice_cae(voice_data_dir=args.data, out_path=args.out)
    elif args.encoder == "denoise_ae":
        train_denoise_ae(voice_data_dir=args.data, out_path=args.out)
    else:
        print("[ERROR] main(): no such encoder {}".format(args.encoder))
        raise


if __name__ == '__main__':
    main()