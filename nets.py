import torch.nn as nn
from config import *

class basis_net(nn.Module):
    def __init__(self, in_size, hid_size, out_size, init, req_grad=True):
        super(basis_net, self).__init__()
        self.lstm1 = nn.LSTM(input_size=in_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hid_size, out_size, batch_first=True)
        # self.lstm1 = nn.LSTM(input_size=in_size, hidden_size=out_size, num_layers=1, batch_first=True)

        if init:
            for p in self.lstm1.parameters():
                # nn.init.normal_(p)
                nn.init.zeros_(p)

            for p in self.lstm2.parameters():
                # nn.init.normal_(p)
                nn.init.zeros_(p)

        if not req_grad:
            for p in self.lstm1.parameters():
                nn.init.zeros_(p)
                p.requires_grad = False

            for p in self.lstm2.parameters():
                nn.init.zeros_(p)
                p.requires_grad = False


    def forward(self, x):
        # Output shape batch_size*seq_length*hidden_size
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x


class gen_net(nn.Module):
    def __init__(self, in_size, hid_size, out_size, init):
        super(gen_net, self).__init__()
        # self.lin = nn.Sequential(nn.Linear(in_features=in_size, out_features=hid_size))
        self.lstm1 = nn.LSTM(input_size=in_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hid_size, out_size, batch_first=True)

        if init:
            for p in self.lstm1.parameters():
                # nn.init.normal_(p)
                nn.init.zeros_(p)
                # p.requires_grad = False

            # for p in self.lstm2.parameters():
                # nn.init.normal_(p)
                # nn.init.zeros_(p)

    def forward(self, x):
        # Output shape batch_size*seq_length*hidden_size
        # x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)
        # out = self.lin(x)
        # anticipated = (x[:, :-1, :] + x[:, 1:, :])/2
        # anticipated = torch.cat((anticipated, x[:, -1, :].unsqueeze(1)), dim=1)
        # out, _ = self.lstm1(anticipated)
        deviate_from_mean = x - x.mean(axis=1).unsqueeze(1)
        out, _ = self.lstm1(deviate_from_mean)
        out, _ = self.lstm2(out)
        return out + x


class target_net(nn.Module):
    def __init__(self, in_size, out_size, init):
        super(target_net, self).__init__()
        self.layers = nn.Sequential(nn.Linear(seq_len*in_size, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 64),
                                    nn.LeakyReLU(),
                                    nn.Linear(64, out_size))
        # self.lstm1 = nn.LSTM(input_size=in_size, hidden_size=32, num_layers=2, batch_first=True)
        # self.lstm2 = nn.LSTM(32, out_size, batch_first=True)

        if init:
            for p in self.layers.parameters():
                # nn.init.normal_(p)
                nn.init.zeros_(p)
                # p.requires_grad = False

            # for p in self.lstm2.parameters():
                # nn.init.normal_(p)
                # nn.init.zeros_(p)

    def forward(self, x):
        # Output shape batch_size*seq_length*hidden_size
        # x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)
        # x = x[:, -1, 0] + 1.0
        x = x.reshape(x.shape[0], -1)
        x = self.layers(x)
        return x.reshape(-1)

