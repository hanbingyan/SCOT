import torch
import numpy as np

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

n_feature = 1
small_batch = 4
seq_len = 20
dt = 1/52
n_iter = 3000
kappa, mu, sigma = 0.5, 0.5, 0.2
V_last = 0.5

v_min = 0.0
v_max = 1.0
torch.manual_seed(12345)
# check gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ss = np.random.SeedSequence(12345)
# child_seeds = ss.spawn(small_batch)
# rng = [np.random.default_rng(s) for s in child_seeds]
rng = np.random.default_rng(12345)


def vol_sim(cu_len=1):
    W = rng.normal(size=(small_batch, seq_len-cu_len+1))
    result = np.zeros((small_batch, seq_len+1))
    result[:, :cu_len] = V_last
    for i in range(cu_len, seq_len+1, 1):
        # result[i] = result[i-1] + sigma*np.sqrt(dt)*W[i-cu_len]
        diffusion = np.multiply(v_max-result[:, i-1], result[:, i-1] - v_min)*dt/(np.sqrt(v_max) - np.sqrt(v_min))**2
        diffusion = np.clip(diffusion, a_min=0.0, a_max=None)
        diffusion = np.sqrt(diffusion)
        result[:, i] = result[:, i-1] + kappa*(mu - result[:, i-1])*dt + sigma*np.multiply(diffusion, W[:, i-cu_len])
    return np.expand_dims(np.around(result[:, 1:], decimals=4), axis=-1)

def f(y):
    # y shape batch*seq_length
    return y.mean(axis=(1,2))

