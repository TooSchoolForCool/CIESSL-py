import os

import librosa
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()


    def save(self, out_path):
        torch.save(self.state_dict(), out_path)


    def load(self, model_dir):
        if torch.cuda.is_available():
            super(AutoEncoder, self).cuda()
            self.load_state_dict( torch.load(model_dir) )
            print("Load Model [{}] to GPU".format(model_dir))
        else:
            self.load_state_dict( torch.load(model_dir, map_location="cpu") )
            print("Load Model [{}] to CPU".format(model_dir))    


class VoiceEncoder(AutoEncoder):
    def __init__(self, nn_structure=[8000*16, 1000, 500, 100]):
        super(VoiceEncoder, self).__init__()

        self.__build_encoder(nn_structure)
        self.__build_decoder(nn_structure[::-1])
        self.loss_ = nn.MSELoss(size_average=False)


    def encode(self, x):
        return self.encoder_(x)


    def decode(self, z):
        return self.decoder_(z)


    def forward(self, x):
        code = self.encode(x)
        out = self.decode(code)
        return out


    def loss(self, recon_x, x):
        return self.loss_(recon_x, x)


    def __build_encoder(self, structure):
        net = []

        # here, we do not handle the very last layer
        for i in range(1, len(structure) - 1):
            net.append( nn.Linear(structure[i - 1], structure[i]) )
            net.append( nn.ReLU() )

        # build last layer
        net.append(nn.Linear(structure[-2], structure[-1]))

        self.encoder_ = nn.Sequential(*net)


    def __build_decoder(self, structure):
        net = []

        # here, we do not handle the very last layer
        for i in range(1, len(structure) - 1):
            net.append( nn.Linear(structure[i - 1], structure[i]) )
            net.append( nn.ReLU() )

        # build last layer
        net.append(nn.Linear(structure[-2], structure[-1]))
        net.append( nn.Sigmoid() )

        self.decoder_ = nn.Sequential(*net)


class VoiceVAE(AutoEncoder):
    def __init__(self, nn_structure=[6000, 2000, 500, 100]):
        super(VoiceVAE, self).__init__()

        self.__build_encoder(nn_structure)
        self.__build_decoder(nn_structure[::-1]) # reverse the structure


    def __build_encoder(self, structure):
        net = []

        # here, we do not handle the very last layer
        for i in range(1, len(structure) - 1):
            net.append( nn.Linear(structure[i - 1], structure[i]) )
            net.append( nn.ReLU() )

        self.encoder_prefix_ = nn.Sequential(*net)

        self.encoder_mu_ = nn.Linear(structure[-2], structure[-1])
        self.encoder_sigma_ = nn.Linear(structure[-2], structure[-1])


    def __build_decoder(self, structure):
        net = []

        # here, we do not handle the very last layer
        for i in range(1, len(structure) - 1):
            net.append( nn.Linear(structure[i - 1], structure[i]) )
            net.append( nn.ReLU() )

        # build last layer
        net.append(nn.Linear(structure[-2], structure[-1]))
        net.append( nn.Sigmoid() )

        self.decoder_ = nn.Sequential(*net)


    def encode(self, x):
        """
        Encode for using the autoencoder
        """
        h1 = self.encoder_prefix_(x)

        mu = self.encoder_mu_(h1)
        logvar = self.encoder_sigma_(h1)

        return self.reparametrize(mu, logvar)


    def encode_(self, x):
        """
        Encode for training
        """
        h1 = self.encoder_prefix_(x)
        
        mu = self.encoder_mu_(h1)
        logvar = self.encoder_sigma_(h1)

        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def decode(self, z):
        return self.decoder_(z)


    def forward(self, x):
        mu, logvar = self.encode_(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


    def loss(self, recon_x, x, mu, logvar):
        """
        Args:
            recon_x: generating images
            x: origin images
            mu: latent mean
            logvar: latent log variance
        """
        reconstruction_function = nn.MSELoss(size_average=False)
        BCE = reconstruction_function(recon_x, x)  # mse loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return BCE + KLD


class VoiceConvAE(AutoEncoder):
    def __init__(self, code_size=256):
        super(VoiceConvAE, self).__init__()
        
        self.encoder_ = nn.Sequential(
            nn.Conv2d(1, 128, (5, 5), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            # nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.Conv2d(128, 256, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.Conv2d(256, 256, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.Conv2d(256, 512, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),

            nn.Conv2d(512, 512, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),

            nn.Conv2d(512, 1024, (2, 2), stride=(2, 2)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(True),

            nn.Conv2d(1024, code_size, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(code_size),
        )

        self.encoder_fc_ = nn.Sequential(
            nn.Linear(2048, 256),
            nn.LeakyReLU(True)
        )

        self.decoder_fc_ = nn.Sequential(
            nn.Linear(256, 2048),
            nn.LeakyReLU(True)
        )

        self.decoder_ = nn.Sequential(
            nn.ConvTranspose2d(code_size, 1024, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(1024, 512, (2, 2), stride=(2, 2)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(512, 512, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(512, 256, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(256, 256, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(256, 128, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(128, 128, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(128, 1, (5, 5), stride=(2, 2)),
            nn.Sigmoid(),
        )

        # define loss function parameters
        mask = self.__create_loss_mask(peak=10.0, hz_flat=4000)[:255]
        mask = np.tile(mask, (255, 1)).T
        mask = torch.Tensor(mask)
        self.loss_weight_mask_ = Variable(mask)
        if torch.cuda.is_available():
            self.loss_weight_mask_ = self.loss_weight_mask_.cuda()


    def encode(self, x):
        conv_code = self.encoder_(x)
        return conv_code

        # conv_code = conv_code.view(conv_code.shape[0], -1)
        # return self.encoder_fc_(conv_code)


    def decode(self, x):
        # dec_code = self.decoder_fc_(x)
        # dec_code = dec_code.view(dec_code.size(0), 128, 4, 4)
        return self.decoder_(x)


    def forward(self, x):
        encode = self.encode(x)
        xhat = self.decode(encode)
        return xhat


    def loss(self, x, recon_x):
        out = (x - recon_x) ** 2        
        out = out * self.loss_weight_mask_.expand_as(out)
        loss = out.sum() # or sum over whatever dimensions
        return loss


    def __create_loss_mask(self, peak=10.0, hz_flat=6000, sr=48000, n_fft=1024):
        n = int(n_fft / 2)
        cutoff = np.where(librosa.core.fft_frequencies(sr=sr, n_fft=n_fft) >= hz_flat)[0][0]
        mask = np.concatenate([np.linspace(peak, 1.0, cutoff), np.ones(n - cutoff)])
        return mask




def test_autoencoder():
    from data_loader import DataLoader

    voice_data_dir = "../../data/active_voice"
    map_data_dir = "../../data/map/bh9f_lab_map.json"
    pos_tf_dir = "../config/bh9f_pos_tf.json"
    dl = DataLoader(voice_data_dir, map_data_dir, pos_tf_dir)

    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3
    n_frames = 18000

    model = VoiceEncoder()

    if torch.cuda.is_available():
        print("[INFO] CUDA is available")
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        for voice in dl.voice_data_iterator(seed=1):
            voice_frames = voice["frames"]
            cnt = 0
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
                optimizer.step()

            cnt += 1

        # ===================log========================
        print('epoch [{}/{}], voice sample:{} loss:{:.4f}'.format(epoch + 1, num_epochs, cnt, loss.data[0]))

    model.save("voice_autoencoder.model")


def test_vae():
    from data_loader import DataLoader

    voice_data_dir = "../../data/active_voice"
    map_data_dir = "../../data/map/bh9f_lab_map.json"
    pos_tf_dir = "../config/bh9f_pos_tf.json"
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
        train_loss = 0
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

        if min_loss < train_loss / cnt:
            min_loss = float(train_loss / cnt)
            print("model saved at: {}".format(voice_vae.model))
            model.save("voice_vae.model")


if __name__ == '__main__':
    # test_autoencoder()
    test_vae()