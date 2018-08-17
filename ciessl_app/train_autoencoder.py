import os
import argparse

import torch

from model.autoencoder import VoiceVAE
from data_loader import DataLoader

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
    parser.add_argument(
        "--encoder",
        dest="encoder",
        type=str,
        required=True,
        help="choose what type of encoder to train"
    )
    
    args = parser.parse_args()

    return args


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

    for epoch in range(num_epochs):
        model.train()
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


def main():
    args = arg_parser()

    if args.encoder == "voice_vae":
        train_voice_vae(voice_data_dir=args.voice, map_data_dir=args.map, 
            pos_tf_dir=args.config, out_path=args.out)
    else:
        print("[ERROR] main(): no such encoder {}".format(args.encoder))
        raise


if __name__ == '__main__':
    main()