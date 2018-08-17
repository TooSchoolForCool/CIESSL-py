import os

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class VoiceEncoder(nn.Module):
    def __init__(self):
        super(VoiceEncoder, self).__init__()

        nn_structure = [18000, 1800, 200]

        self.__build_encoder(nn_structure)
        self.__build_decoder(nn_structure[::-1])

    def encode(self, x):
        return self.encoder_(x)


    def decode(self, z):
        return self.decoder_(z)


    def forward(self, x):
        code = self.encode(x)
        out = self.decode(code)
        return out


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
        net.append( nn.Tanh() )

        self.decoder_ = nn.Sequential(*net)


    def save(self, out_path):
        torch.save(self.state_dict(), out_path)


    def load(self, model_dir):
        if torch.cuda.is_available():
            self.load_state_dict( torch.load(model_dir) )
        else:
            self.load_state_dict( torch.load(model_dir, map_location="cpu") )


class VoiceVAE(nn.Module):
    def __init__(self):
        super(VoiceVAE, self).__init__()

        self.fc1 = nn.Linear(18000, 1800)
        self.fc21 = nn.Linear(1800, 200)
        self.fc22 = nn.Linear(1800, 200)

        self.fc3 = nn.Linear(200, 1800)
        self.fc4 = nn.Linear(1800, 18000)


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return nn.Tanh( self.fc4(h3) )


    def forward(self, x):
        mu, logvar = self.encode(x)
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


    def save(self, out_path):
        torch.save(self.state_dict(), out_path)


    def load(self, model_dir):
        if torch.cuda.is_available():
            self.load_state_dict( torch.load(model_dir) )
        else:
            self.load_state_dict( torch.load(model_dir, map_location="cpu") )


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