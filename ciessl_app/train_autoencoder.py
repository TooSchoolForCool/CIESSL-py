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
        "--map",
        dest="map",
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
        "--out",
        dest="out",
        type=str,
        required=True,
        help="output directory"
    )

    valid_encoder = ["voice_vae", "voice_simple", "all_ch_vae", "all_ch_simple"]
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

    def __init__(self, voice_data_dir, map_data_dir, pos_tf_dir, normalize=False):
        super(BatchLoader, self).__init__(voice_data_dir, map_data_dir, pos_tf_dir, 
            is_normalize=normalize)


    def load_batch(self, batch_size, n_frames, suffle=True):
        seed = random.randint(0, 2 ** 32 - 1) if suffle else 0
        batch_data = []

        for voice in super(BatchLoader, self).voice_data_iterator(seed=seed):
            # construct training data 1D array [ch1_sig, ch2_sig, ...]
            frames = voice["frames"].T  # frames (n_channels, n_samples)
            frames = frames[:, :n_frames].flatten()
            batch_data.append(frames)

            if len(batch_data) == batch_size:
                yield np.asarray(batch_data)
                batch_data = []

        if batch_data:
            yield np.asarray(batch_data)


def train_voice_vae(voice_data_dir, map_data_dir, pos_tf_dir, out_path):
    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir)

    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-5
    n_frames = 18000

    model = VoiceVAE()
    if torch.cuda.is_available():
        print("[INFO] CUDA is available")
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    min_loss = 99999999.0

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        voice_cnt = 0
        cnt = 0

        for voice in dl.voice_data_iterator(seed=1):
            voice_frames = voice["frames"]

            for i in range(0, voice_frames.shape[1]):
                frames = torch.Tensor(voice_frames[:n_frames, i])
                frames = Variable(frames)
                if torch.cuda.is_available():
                    frames = frames.cuda()

                optimizer.zero_grad()
                recon_batch, mu, logvar = model(frames)
                loss = model.loss(recon_batch, frames, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                cnt += 1

            voice_cnt += 1
            print('voice sample:{} Average loss: {:.4f}'.format(voice_cnt, 1.0 * train_loss / cnt))
        
        print('====> Epoch: {} voice sample:{} Average loss: {:.4f}'.format(
            epoch, voice_cnt, 1.0 * train_loss / cnt))

        if min_loss > train_loss / cnt:
            min_loss = float(1.0 * train_loss / cnt)
            print("model saved at: {}".format(out_path))
            model.save(out_path)


def train_simple_voice_enc(voice_data_dir, map_data_dir, pos_tf_dir, out_path):
    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir)

    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-5
    n_frames = 18000

    model = VoiceEncoder()

    if torch.cuda.is_available():
        print("[INFO] CUDA is available")
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    min_loss = 99999999.0

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        voice_cnt = 0
        cnt = 0

        for voice in dl.voice_data_iterator(seed=1):
            voice_frames = voice["frames"]
            for i in range(0, voice_frames.shape[1]):
                frames = torch.Tensor(voice_frames[:n_frames, i])
                frames = Variable(frames)
                if torch.cuda.is_available():
                    frames = frames.cuda()
                # ===================forward=====================
                output = model.forward(frames)
                loss = criterion(output, frames)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                cnt += 1

            voice_cnt += 1
            print('voice sample:{} Average loss: {:.4f}'.format(voice_cnt, 1.0 * train_loss / cnt))

        # ===================log========================
        print('====> Epoch: {} voice sample:{} Average loss: {:.4f}'.format(
            epoch, voice_cnt, 1.0 * train_loss / cnt))

        if min_loss > train_loss / cnt:
            min_loss = float(1.0 * train_loss / cnt)
            print("model saved at: {}".format(out_path))
            model.save(out_path)


def train_all_ch_vae(voice_data_dir, map_data_dir, pos_tf_dir, out_path):
    dl = BatchLoader(voice_data_dir, map_data_dir, pos_tf_dir)

    num_epochs = 50000
    batch_size = 8
    learning_rate = 1e-4
    next_lr_descent = 500
    n_frames = 6000
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
        cnt = 0

        if epoch == next_lr_descent:
            learning_rate /= 1.5
            next_lr_descent += 500
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for data in dl.load_batch(batch_size=batch_size, n_frames=n_frames):
            data = torch.Tensor(data)
            data = Variable(data)
            if torch.cuda.is_available():
                data = data.cuda()

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            cnt += 1
            # print('voice sample:{} Average loss: {:.4f}'.format(cnt, 1.0 * train_loss / cnt))

        print('====> Epoch: {} Average loss: {:.7f} learning_rate: {:.2f}'.format(
            epoch, 1.0 * train_loss / cnt, learning_rate))

        if epoch % save_frequency == 0 and min_loss > train_loss / cnt:
            min_loss = float(1.0 * train_loss / cnt)
            print("model saved at: {}".format(out_path))
            model.save(out_path)


def train_all_ch_simple(voice_data_dir, map_data_dir, pos_tf_dir, out_path):
    dl = BatchLoader(voice_data_dir, map_data_dir, pos_tf_dir, normalize=True)

    num_epochs = 50000
    batch_size = 8
    learning_rate = 1e-5
    n_frames = 6000
    save_frequency = 100

    model = VoiceEncoder()
    if torch.cuda.is_available():
        print("[INFO] CUDA is available")
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    min_loss = 99999999.0

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        cnt = 0

        for data in dl.load_batch(batch_size=batch_size, n_frames=n_frames):
            data = torch.Tensor(data)
            data = Variable(data)
            if torch.cuda.is_available():
                data = data.cuda()
            # ===================forward====================
            output = model.forward(data)
            loss = criterion(output, data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            cnt += 1
            # print('voice sample:{} Average loss: {:.4f}'.format(cnt, 1.0 * train_loss / cnt))

        print('====> Epoch: {} Average loss: {:.7f}'.format(
            epoch, 1.0 * train_loss / cnt))

        if epoch % save_frequency == 0 and min_loss > train_loss / cnt:
            min_loss = float(1.0 * train_loss / cnt)
            print("model saved at: {}".format(out_path))
            model.save(out_path)


def main():
    args = arg_parser()

    if args.encoder == "voice_vae":
        train_voice_vae(voice_data_dir=args.voice, map_data_dir=args.map, 
            pos_tf_dir=args.config, out_path=args.out)
    elif args.encoder == "voice_simple":
        train_simple_voice_enc(voice_data_dir=args.voice, map_data_dir=args.map, 
            pos_tf_dir=args.config, out_path=args.out)
    elif args.encoder == "all_ch_vae":
        train_all_ch_vae(voice_data_dir=args.voice, map_data_dir=args.map, 
            pos_tf_dir=args.config, out_path=args.out)
    elif args.encoder == "all_ch_simple":
        train_all_ch_simple(voice_data_dir=args.voice, map_data_dir=args.map, 
            pos_tf_dir=args.config, out_path=args.out)
    else:
        print("[ERROR] main(): no such encoder {}".format(args.encoder))
        raise


if __name__ == '__main__':
    main()