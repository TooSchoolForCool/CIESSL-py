import os
import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from model.autoencoder import VoiceVAE, VoiceEncoder
from model.data_loader import DataLoader


def arg_parser():
    """Argument Parser

    Parse arguments from command line, and perform error checking

    Returns:
        An argument object which contains arguments from cmd line
    """
    parser = argparse.ArgumentParser(prog='Training Voice AutoEncoder')

    parser.add_argument(
        "--voice",
        dest="voice",
        type=str,
        required=True,
        help="Input voice data directory"
    )

    parser.add_argument(
        "--out",
        dest="out",
        type=str,
        required=True,
        help="output directory"
    )

    valid_encoder = ["voice_vae", "voice_ae"]
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


class BatchLoader(DataLoader):

    def __init__(self, data_dir, mode="all"):
        self.__load_dataset(data_dir, mode)


    def __load_dataset(self, data_dir, mode):
        dataset = []

        for file in os.listdir(data_dir):
            if file.endswith(".pickle"):
                data = np.load( os.path.join(data_dir, file) )
                dataset.append(data)

        if mode == "train":
            idx = np.arange( len(dataset) )
            rs = np.random.RandomState(0)
            rs.shuffle(idx)
            dataset = np.asarray(dataset)[idx]

        self.dataset_ = dataset


    def size(self):
        return len( self.dataset_ )


    def load_batch(self, batch_size, suffle=True, flatten=True):
        skip_n_frames = 2000
        n_frames = 4000

        idx = np.arange( len(self.dataset_) )

        if suffle:
            rs = np.random.RandomState( random.randint(0, 2 ** 32 - 1) )
            rs.shuffle(idx)
        
        batch_data = []
        for i in idx:
            data = self.dataset_[i][skip_n_frames:skip_n_frames+n_frames, :]
    
            if flatten:
                data = data.T.flatten()

            batch_data.append(data)

            if len(batch_data) == batch_size:
                # (n_samples, n_features)
                yield np.asarray(batch_data)
                batch_data = []

        if batch_data:
            yield np.asarray(batch_data)


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

        if epoch % save_frequency == 0 and min_loss > train_loss / bl.size():
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


def main():
    args = arg_parser()

    if args.encoder == "voice_vae":
        train_voice_vae(voice_data_dir=args.voice, out_path=args.out)
    elif args.encoder == "voice_ae":
        train_voice_ae(voice_data_dir=args.voice, out_path=args.out)
    else:
        print("[ERROR] main(): no such encoder {}".format(args.encoder))
        raise


if __name__ == '__main__':
    main()