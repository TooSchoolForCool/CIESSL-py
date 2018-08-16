import os

import torch
from torch import nn
from torch.autograd import Variable


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
    nn_structure = [18000, 12000, 6000, 2000, 500]

    model = AutoEncoder(nn_structure=nn_structure)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        for voice in dl.voice_data_iterator(seed=1):
            voice_frames = voice["frames"]
            for i in range(0, voice_frames.shape[1]):
                frames = torch.Tensor(voice_frames[:n_frames, i])
                frames = Variable(frames)
                # ===================forward=====================
                output = model.forward(frames)
                loss = criterion(output, frames)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0]))

    torch.save(model.state_dict(), './voice_autoencoder.pth')


if __name__ == '__main__':
    test_autoencoder()