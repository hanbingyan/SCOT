import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

import randy
import dety
from nets import gen_net, target_net
from sp_config import *

np.random.seed(12345)

SP = pd.read_csv('SP_Sub.csv')
SP['Volume'] *= 10
SP['mom1'] *= 10
SP['mom3'] = SP['mom3']*10/3
SP['ROC_15'] = SP['ROC_15']/15*10
SP['EMA_20'] = SP['EMA_20']/1000/20
SP['Oil'] *= 10
SP['Gold'] *= 10
SP['DAAA'] /= 10
SP['AAPL'] *= 10
SP['JNJ'] *= 10
SP['XOM'] *= 10
SP['TE1'] /= 10
Date = SP['Date']
SP = SP.iloc[:, 1:].to_numpy()
max_len = SP.shape[0]

obj_f = target_net(in_size=in_feature*seq_len, out_size=1, init=False).to(device)
gen_Y = gen_net(in_size=in_feature, hid_size=4, out_size=in_feature, init=True).to(device)

criterion = nn.MSELoss()
tar_opt = optim.Adam(list(obj_f.parameters()), lr = 5e-3)
gen_opt = optim.Adam(list(gen_Y.parameters()), lr = 1e-4)

def SP_sampler(bat_size, end_idx, hi=100, rand=True):
    res = []
    if rand:
        for _ in range(bat_size):
            idx = np.random.randint(low=0, high=hi, size=1, dtype=int)[0]
            x = SP[end_idx-idx-seq_len-1 : end_idx-idx, :].copy()
            x[:, 0] /= x[0, 0]
            x = x.reshape(seq_len+1, in_feature)
            res.append(x.copy())
    else:
        for idx in range(100):
            x = SP[end_idx+idx : end_idx+idx+seq_len+1, :].copy()
            x[:, 0] /= x[0, 0]
            x = x.reshape(seq_len + 1, in_feature)
            res.append(x.copy())
    return np.array(res)

sep_date = 1100
print('The separate date is', Date[sep_date])
print('Data from 100+seq days before this date are for training')
print('Data from 100+seq days AFTER this date are for out-of-sample testing')

for ind in range(800):
    x_tar = SP_sampler(small_batch, end_idx=sep_date)
    input_x = x_tar[:, :-1, :]
    tar = torch.tensor(np.log(x_tar[:, -1, 0]), dtype=torch.float, device=device)
    in_x = torch.tensor(input_x, dtype=torch.float, device=device)
    y = obj_f(in_x)
    loss = criterion(y, tar)
    obj_f.zero_grad()
    loss.backward()
    tar_opt.step()

    if ind%20 == 0:
        print('loss', loss.item())

for ind in range(800):
    x_tar = SP_sampler(small_batch, end_idx=sep_date)
    input_x = x_tar[:, :-1, :]
    in_x = torch.tensor(input_x, dtype=torch.float, device=device)

    gen_loss = criterion(gen_Y(in_x), in_x)
    gen_Y.zero_grad()
    gen_loss.backward()
    gen_opt.step()

    if ind%20 == 0:
        print('gen loss', gen_loss.item())

for param in list(obj_f.parameters()):
    param.requires_grad = False

# to reproduce the results in the paper, run each algo separately
dety.train(SP_sampler, obj_f, idx=sep_date, causal=False)
# randy.train(SP_sampler, gen_Y, obj_f, idx=sep_date, causal=False)
