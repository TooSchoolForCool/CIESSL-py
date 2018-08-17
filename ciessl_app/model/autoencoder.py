import os

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, nn_structure):
        super(AutoEncoder, self).__init__()

        try:
            assert(len(nn_structure) > 1)
        except:
            print("[ERROR] AutoEncoder.__init__(): nn_structure size must be greater than 1, " 
                + "current length is {}".format(len(nn_structure)))
            raise

        self.__build_encoder(nn_structure)
        self.__build_decoder(nn_structure[::-1])


    def forward(self, x):
        code = self.encoder_(x)
        out = self.decoder_(code)
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
        self.load_state_dict( torch.load(model_dir) )


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(18000, 9000)
        self.fc2 = nn.Linear(9000, 3000)
        self.fc31 = nn.Linear(3000, 300)
        self.fc32 = nn.Linear(3000, 300)

        self.fc4 = nn.Linear(300, 3000)
        self.fc5 = nn.Linear(3000, 9000)
        self.fc6 = nn.Linear(9000, 18000)


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))

        return torch.sigmoid( self.fc6(h5) )


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
        self.load_state_dict( torch.load(model_dir) )


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
    nn_structure = [18000, 9000, 3000, 300]

    model = AutoEncoder(nn_structure=nn_structure)

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
    learning_rate = 1e-3
    n_frames = 18000

    model = VAE()
    if torch.cuda.is_available():
        print("[INFO] CUDA is available")
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
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

        print('====> Epoch: {} voice sample:{} Average loss: {:.4f}'.format(
            epoch, cnt, 1.0 * train_loss / cnt))

    model.save("voice_vae.model")


if __name__ == '__main__':
    test_autoencoder()
    # test_vae()