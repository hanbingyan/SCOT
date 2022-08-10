# Real parameter given
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from datetime import datetime
from vol_config import *

def cost(x, y):
    return (x - y).pow(2).sum(axis=(1, 2))/x.shape[0]

def beta(x, lam):
    y = (x + torch.rand_like(x) * 1e-3).clone().requires_grad_()
    t = 0
    b_old = 10
    b = f(y).sum() - lam*cost(x, y).sum()

    while t<40 and torch.abs((b - b_old)/b_old) > 1e-3:
        b_old = b.item()
        t += 1
        b.backward(retain_graph=True)
        y = torch.clamp(y + 50/(t+1)*y.grad, min=0.0, max=3.0).clone().detach().requires_grad_()
        b = f(y).sum() - lam * cost(x, y).sum()
    return y


####################### Training ##################################

def train(args):
    radius = args.radius
    lam_init = args.lam_init
    dual_hist = torch.zeros(n_iter, device='cpu')
    lam_hist = torch.zeros(n_iter, device='cpu')
    cost_hist = torch.zeros(n_iter, device='cpu')
    sam_mean = torch.zeros(n_iter, device='cpu')
    lam = torch.tensor([lam_init], requires_grad=True, device=device)
    velocity = 0.0

    # load volatility data
    with open('x52.pickle', 'rb') as fp:
        x_tar = pickle.load(fp)
    x = torch.tensor(x_tar, dtype=torch.float, device=device)

    for iter in range(n_iter):

        y = beta(x, lam.detach())
        b = f(y.detach()).sum() - lam * cost(x, y.detach()).sum()
        dual = lam * radius + b/batch
        obj = dual

        lam.grad = None
        obj.backward()
        D_lam = lam.grad

        velocity = 0.9*velocity - 1/(0.1*iter+10)*D_lam
        lam = torch.clamp(lam + velocity, min=0.0).clone().detach().requires_grad_()

        dual_hist[iter] = dual.item()
        lam_hist[iter] = lam.item()
        cost_hist[iter] = cost(x, y).mean().item()

        sam_mean[iter] = f(y).mean().item()
        if iter % 100 == 0:
            print('iter', iter, 'dual', dual.item(), 'lam', lam.item(),
                  'f mean', f(y).mean().item(), 'cost mean', cost(x, y).mean().item())

    x_last = x[-1, :, :].reshape(-1).cpu().numpy()
    y_last = y[-1, :, :].reshape(-1).detach().cpu().numpy()

    return dual_hist.numpy(), lam_hist.numpy(), sam_mean.numpy(), x_last, y_last, cost_hist.numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Volatility estimation')
    parser.add_argument('--radius', type=float, default=0.3, help='Wasserstein ball radius')
    parser.add_argument('--lam_init', type=float, default=10.0, help='Initial value of lambda')
    args = parser.parse_args()

    dual_runs = []
    lam_runs = []
    sam_runs = []
    cost_runs = []
    x_hist = []
    y_hist = []

    for _ in range(n_runs):
        dual_hist, lam_hist, sam_mean, x_last, y_last, cost_hist = train(args)
        dual_runs.append(dual_hist)
        lam_runs.append(lam_hist)
        sam_runs.append(sam_mean)
        x_hist.append(x_last)
        y_hist.append(y_last)
        cost_runs.append(cost_hist)

    dual_runs = np.array(dual_runs)
    lam_runs = np.array(lam_runs)
    sam_runs = np.array(sam_runs)
    cost_runs = np.array(cost_runs)
    x_hist = np.array(x_hist)
    y_hist = np.array(y_hist)

    sub_folder = '{}_{}.{}'.format('vol_ot', datetime.now().strftime('%H'), datetime.now().strftime('%M'))
    log_dir = './logs/{}'.format(sub_folder)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save params configuration
    with open('{}/params.txt'.format(log_dir), 'w') as fp:
        fp.write('Params setting \n')
        fp.write('batch size {}, seq_len {}, n_iter {} \n'.format(batch, seq_len, n_iter))
        fp.write('radius: {} \n'.format(args.radius))
        fp.write('lambda init: {} \n'.format(args.lam_init))

    with open('{}/dual.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(dual_runs, fp)

    with open('{}/sam.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(sam_runs, fp)

    with open('{}/lam.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(lam_runs, fp)

    with open('{}/cost.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(cost_runs, fp)

    with open('{}/x_hist.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(x_hist, fp)

    with open('{}/y_hist.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(y_hist, fp)
