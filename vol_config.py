import torch
import numpy as np
from nets import vol_obj

class bcolors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PINK = '\033[35m'
    GREY = '\033[36m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

# number of runs
n_runs = 50
n_feature = 1
seq_len = 52
n_iter = 4000
batch = 100

torch.manual_seed(12345)
# check gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rng = np.random.default_rng(12345)

f = vol_obj(in_size=seq_len, out_size=1).to(device)

