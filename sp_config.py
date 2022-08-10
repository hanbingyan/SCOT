import torch

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

torch.manual_seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

small_batch = 32
seq_len = 20
n_iter = 2000

radius = 0.4
lam_init = 1.0

in_feature = 16
out_feature = 1

iter_const = 200
entropy_const = 0.01