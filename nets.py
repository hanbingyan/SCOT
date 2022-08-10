import torch
import torch.nn as nn

class basis_net(nn.Module):
    def __init__(self, in_size, hid_size, out_size, init, req_grad=True):
        super(basis_net, self).__init__()
        self.lstm1 = nn.LSTM(input_size=in_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hid_size, out_size, batch_first=True)

        if init:
            for p in self.lstm1.parameters():
                nn.init.zeros_(p)

            for p in self.lstm2.parameters():
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
        self.lstm1 = nn.LSTM(input_size=in_size, hidden_size=hid_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hid_size, out_size, batch_first=True)

        if init:
            for p in self.lstm1.parameters():
                nn.init.zeros_(p)


    def forward(self, x):
        deviate_from_mean = x - x.mean(axis=1).unsqueeze(1)
        out, _ = self.lstm1(deviate_from_mean)
        out, _ = self.lstm2(out)
        return out + x


class target_net(nn.Module):
    def __init__(self, in_size, out_size, init):
        super(target_net, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_size, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 64),
                                    nn.LeakyReLU(),
                                    nn.Linear(64, out_size))
        if init:
            for p in self.layers.parameters():
                nn.init.zeros_(p)


    def forward(self, x):
        # Output shape batch_size*seq_length*hidden_size
        x = x.reshape(x.shape[0], -1)
        x = self.layers(x)
        return x.reshape(-1)

class vol_obj(nn.Module):
    def __init__(self, in_size, out_size):
        super(vol_obj, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_size, out_size),
                                    nn.ReLU())

        for p in self.layers.parameters():
            nn.init.uniform_(p, 0.0, 0.1)
            p.requires_grad = False

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.layers(x)
        return x.reshape(-1)

class vol_gen(nn.Module):
    def __init__(self, in_size, out_size):
        super(vol_gen, self).__init__()
        self.lstm1 = nn.LSTM(input_size=in_size, hidden_size=4, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(4, out_size, batch_first=True)

    def forward(self, x):
        y, _ = self.lstm1(x)
        y, _ = self.lstm2(y)
        return torch.exp(y)
