# Volatility calibration
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from datetime import datetime
from nets import basis_net, vol_gen
from vol_config import *
from utils import *

def c(x, y, p=2):
    '''
    L2 distance between vectors, using expanding and hence is more memory intensive
    :param x: x is tensor of shape [batch_size, time steps, features]
    :param y: y is tensor of shape [batch_size, time steps, features]
    :param p: power
    :return: cost matrix: a matrix of size [batch_size, batch_size]
    '''
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    b = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    c = torch.sum(b, -1)
    return c/x.shape[0]

def train(gen_Y, f, args):
    radius = args.radius
    lam_init = args.lam_init
    causal =args.causal

    test_H = basis_net(in_size=n_feature, hid_size=4, out_size=n_feature, init=False, req_grad=causal).to(device)
    test_M = basis_net(in_size=n_feature, hid_size=4, out_size=n_feature, init=False, req_grad=causal).to(device)
    var_hist = torch.zeros(n_iter, device='cpu')
    lam_hist = torch.zeros(n_iter, device='cpu')
    cost_hist = torch.zeros(n_iter, device='cpu')
    f_mean = torch.zeros(n_iter, device='cpu')

    lam = torch.tensor([lam_init], requires_grad=True, device=device)

    optimMH = optim.Adam(list(test_M.parameters()) + list(test_H.parameters()), lr = 5e-2)
    optimY = optim.Adam(list(gen_Y.parameters()), lr = 1e-2)
    velocity = 0.0

    # load volatility data
    with open('x52.pickle', 'rb') as fp:
        x_tar = pickle.load(fp)
    x = torch.tensor(x_tar, dtype=torch.float, device=device)

    for iter in range(n_iter):

        ## Inner minimization over lambda, h, g
        g = test_M(x)
        y = gen_Y(x)
        h = test_H(y.detach())[:, :-1, :]*(1 + 1000/(1 + 0.01*iter))
        wass_sam, pi = compute_sinkhorn(x, y.detach(), h, g, lam, f, c)
        in_loss = lam * radius + wass_sam + martingale_regularization(g)
        test_H.zero_grad()
        test_M.zero_grad()
        lam.grad = None
        in_loss.backward()
        D_lam = lam.grad
        optimMH.step()
        velocity = 0.9*velocity - 1/(0.1*iter + 10)*D_lam
        lam = torch.clamp(lam + velocity, min=0.0).clone().detach().requires_grad_()

        with torch.no_grad():
            for param in test_M.parameters():
                param.clamp_(-50.0, 50.0)
            for param in test_H.parameters():
                param.clamp_(-50.0, 50.0)

        #### Maximization over Generator ######
        g = test_M(x).detach()
        y = gen_Y(x)
        h = test_H(y)[:, :-1, :]*(1 + 1000/(1 + 0.01*iter))
        out_wass, out_pi = compute_sinkhorn(x, y, h, g, lam.detach(), f, c)
        out_loss = - out_wass

        gen_Y.zero_grad()
        out_loss.backward()
        optimY.step()

        # calculate H*M
        DeltaM = g[:, 1:, :] - g[:, :-1, :]
        time_steps = h.shape[1]
        sum_over_j = torch.sum(h[:, None, :, :] * DeltaM[None, :, :, :], -1)
        C_hM = torch.sum(sum_over_j, -1) / time_steps
        HMPi = torch.sum(C_hM*out_pi)

        var_hist[iter] = -out_loss.item() + lam.item()*radius
        lam_hist[iter] = lam.item()
        cost_hist[iter] = torch.sum(c(x,y)*out_pi).item()
        f_mean[iter] = f(y).mean().item()

        if iter % 100 == 0:
            print('iter', iter, 'dual', var_hist[iter].item(), 'f(y)', f(y).mean().item(),
                  'HMPi', HMPi.mean().item(), 'lam', lam.item(), 'cost', torch.sum(c(x, y) * out_pi).item())


    x_last = x[-1, :, :].reshape(-1).cpu().numpy()
    y_last = y[-1, :, :].reshape(-1).detach().cpu().numpy()

    return var_hist.numpy(), lam_hist.numpy(), cost_hist.numpy(), f_mean.numpy(), x_last, y_last

############## Main #######################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Volatility estimation')
    parser.add_argument('--radius', type=float, default=0.3, help='Wasserstein ball radius')
    parser.add_argument('--lam_init', type=float, default=10.0, help='Initial value of lambda')
    parser.add_argument('--cot', dest='causal', action='store_true')
    parser.add_argument('--ot', dest='causal', action='store_false')
    parser.set_defaults(causal=False)
    args = parser.parse_args()


    dual_runs = []
    lam_runs = []
    cost_runs = []
    f_runs = []

    x_hist = []
    y_hist = []

    for _ in range(n_runs):
        gen_Y = vol_gen(in_size=n_feature, out_size=n_feature).to(device)
        var_hist, lam_hist, cost_hist, f_mean, x_last, y_last = train(gen_Y, f, args)
        dual_runs.append(var_hist)
        lam_runs.append(lam_hist)
        cost_runs.append(cost_hist)
        f_runs.append(f_mean)
        x_hist.append(x_last)
        y_hist.append(y_last)

    dual_runs = np.array(dual_runs)
    lam_runs = np.array(lam_runs)
    cost_runs = np.array(cost_runs)
    f_runs = np.array(f_runs)
    x_hist = np.array(x_hist)
    y_hist = np.array(y_hist)

    sub_folder = '{}_{}_{}.{}'.format('vol_str', args.causal,
                                      datetime.now().strftime('%H'), datetime.now().strftime('%M'))

    log_dir = './logs/{}'.format(sub_folder)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save params configuration
    with open('{}/params.txt'.format(log_dir), 'w') as fp:
        fp.write('Params setting \n')
        fp.write('COT or not: {} \n'.format(args.causal))
        fp.write('batch size {}, seq_len {}, n_iter {} \n'.format(batch, seq_len, n_iter))
        fp.write('Radius: {} \n'.format(args.radius))
        fp.write('Lambda Init: {} \n'.format(args.lam_init))

    with open('{}/dual.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(dual_runs, fp)

    with open('{}/sam.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(f_runs, fp)

    with open('{}/lam.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(lam_runs, fp)

    with open('{}/cost.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(cost_runs, fp)

    with open('{}/x_hist.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(x_hist, fp)

    with open('{}/y_hist.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(y_hist, fp)
