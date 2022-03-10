# Real parameter given
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from datetime import datetime
from nets import basis_net
from volatility_utils import *
from utils import *


class CIR_Y(nn.Module):
    def __init__(self):
        super(CIR_Y, self).__init__()
        self.kappa = nn.Parameter(torch.tensor([kappa]))
        self.mu = nn.Parameter(torch.tensor([0.0]))
        self.sigma = nn.Parameter(torch.tensor([sigma]))
        self.prob = 0.5

    def forward(self, x):
        tol_mu = self.mu + x.mean(axis=1).reshape(-1)
        W = torch.randn((small_batch, seq_len), device=device)
        result = torch.zeros((small_batch, seq_len+1), device=device)
        result[:, 0] = V_last
        for i in range(1, seq_len+1, 1):
            # result[i] = sigma*np.sqrt(dt)*W[i-cu_len]
            result[:, i] = result[:, i - 1] + self.kappa*(tol_mu - result[:, i - 1])*dt + \
                           self.sigma*torch.mul(torch.sqrt(torch.clamp(result[:, i - 1]*dt, min=0.0, max=None)), W[:, i - 1])
        return self.prob*result[:, 1:].unsqueeze(-1) + (1 - self.prob)*x.mean(axis=1).unsqueeze(1)


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
    return c

def train(gen_Y, f, args):
    radius = args.radius
    lam_init = args.lam_init
    causal =args.causal

    test_H = basis_net(in_size=n_feature, hid_size=4, out_size=n_feature, init=False, req_grad=causal).to(device)
    test_M = basis_net(in_size=n_feature, hid_size=4, out_size=n_feature, init=False, req_grad=causal).to(device)
    var_hist = torch.zeros(n_iter, device=device)
    lam_hist = torch.zeros(n_iter, device=device)
    cost_hist = torch.zeros(n_iter, device=device)
    kappa_hist = torch.zeros(n_iter, device=device)
    mu_hist = torch.zeros(n_iter, device=device)
    sigma_hist = torch.zeros(n_iter, device=device)
    # prob_hist = torch.zeros(n_iter, device=device)

    f_mean = torch.zeros(n_iter, device=device)
    lam = torch.tensor([lam_init], requires_grad=True, device=device)

    optimMH = optim.Adam(list(test_M.parameters()) + list(test_H.parameters()), lr = 5e-2) # 1e-2
    optimY = optim.Adam(list(gen_Y.parameters()), lr = 1e-2)
    velocity = 0.0

    for iter in range(n_iter):

        ## Inner minimization over lambda, h, g
        # sample from reference distribution
        # sim_x = pool.map(simx_helper, [k for k in range(small_batch)])
        x_tar = vol_sim()
        x = torch.tensor(x_tar, dtype=torch.float, device=device)

        g = test_M(x)
        y = gen_Y(x).detach()
        h = test_H(y)[:, :-1, :]
        wass_sam, pi = compute_sinkhorn(x, y, h, g, lam, f, c)
        in_loss = lam * radius + wass_sam + martingale_regularization(g)
        test_H.zero_grad()
        test_M.zero_grad()
        lam.grad = None
        in_loss.backward()
        D_lam = lam.grad
        optimMH.step()
        velocity = 0.9*velocity - 1/(0.1*iter + 200)*D_lam
        lam = torch.clamp(lam + velocity, min=0.0).clone().detach().requires_grad_()

        with torch.no_grad():
            for param in test_M.parameters():
                # param.add_(torch.randn(param.size(), device=device)/10)
                param.clamp_(-5.0, 5.0)
            for param in test_H.parameters():
                # param.add_(torch.randn(param.size(), device=device)/10)
                param.clamp_(-5.0, 5.0)
            gen_Y.state_dict()['sigma'].clamp_(0.1, 0.5)
            # gen_Y.state_dict()['prob'].clamp_(0.0001, 0.9999)

        #### Maximization over Generator ######
        # sample from reference distribution
        x_tar = vol_sim()
        x = torch.tensor(x_tar, dtype=torch.float, device=device)
        g = test_M(x).detach()
        y = gen_Y(x)



        h = test_H(y)[:, :-1, :].detach()
        out_wass, out_pi = compute_sinkhorn(x, y, h, g, lam.detach(), f, c)
        out_loss = - out_wass

        # calculate H*M
        DeltaM = g[:, 1:, :] - g[:, :-1, :]
        # ht = h[:, :-1, :]
        time_steps = h.shape[1]
        sum_over_j = torch.sum(h[:, None, :, :] * DeltaM[None, :, :, :], -1)
        C_hM = torch.sum(sum_over_j, -1) / time_steps
        HMPi = torch.sum(C_hM*out_pi)

        gen_Y.zero_grad()
        out_loss.backward()
        optimY.step()

        var_hist[iter] = -out_loss.item() + lam.item()*radius
        lam_hist[iter] = lam.item()
        cost_hist[iter] = torch.sum(c(x,y)*out_pi).item()
        f_mean[iter] = f(x).mean().item()

        kappa_hist[iter] = gen_Y.state_dict()['kappa'].item()
        mu_hist[iter] = gen_Y.state_dict()['mu'].item()
        sigma_hist[iter] = gen_Y.state_dict()['sigma'].item()
        # prob_hist[iter] = gen_Y.state_dict()['prob'].item()

        if iter % 100 == 0:
            print('iter', iter, 'dual', var_hist[iter].item(), 'f(y)', f(y).mean().item(), 'HMPi', HMPi.mean().item(),
                  'lam', lam.item(), 'cost', torch.sum(c(x,y)*out_pi).item())
            for name, param in gen_Y.named_parameters():
                print(name, param)

    # plt.plot(var_hist.numpy())
    # plt.show()

    # plt.plot(lam_hist.numpy())
    # plt.show()

    x_last = x[-1, :, :].reshape(-1).numpy()
    y_last = y[-1, :, :].reshape(-1).detach().numpy()

    # plt.plot(y_last.reshape(-1).detach().numpy(), label='y')
    # plt.plot(x_last.reshape(-1).numpy(), label='x')
    # plt.legend(loc='best')
    # plt.show()

    return var_hist.numpy(), lam_hist.numpy(), cost_hist.numpy(), kappa_hist.numpy(), \
           mu_hist.numpy(), sigma_hist.numpy(), f_mean.numpy(), x_last, y_last

############## Main #######################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Volatility estimation')
    parser.add_argument('--radius', type=float, default=1.0, help='Wasserstein ball radius')
    parser.add_argument('--lam_init', type=float, default=10.0, help='Initial value of lambda')
    parser.add_argument('--cot', dest='causal', action='store_true')
    parser.add_argument('--ot', dest='causal', action='store_false')
    parser.set_defaults(causal=True)
    args = parser.parse_args()


    #### Pretrain netY to make it similar to reference measure #########
    # gen_Y = CIR_Y().to(device)
    # criterion = nn.MSELoss()
    # init_opt = optim.Adam(list(gen_Y.parameters()), lr = 5e-3)
    # for ind in range(200):
    #     x_cpu = vol_sim()
    #     x = torch.tensor(x_cpu, dtype=torch.float, device=device)
    #     y = gen_Y(x)
    #     loss = criterion(y, x)
    #     gen_Y.zero_grad()
    #     loss.backward()
    #     init_opt.step()
    #     if ind%20 == 0:
    #         print('loss', loss.item())

    # number of runs
    n_runs = 50
    dual_runs = []
    lam_runs = []
    cost_runs = []
    kappa_runs = []
    mu_runs = []
    sigma_runs = []
    # prob_runs = []
    f_runs = []

    x_hist = []
    y_hist = []

    for _ in range(n_runs):
        gen_Y = CIR_Y().to(device)
        var_hist, lam_hist, cost_hist, kappa_hist, mu_hist, sigma_hist, f_mean, x_last, y_last = train(gen_Y, f, args)
        dual_runs.append(var_hist)
        lam_runs.append(lam_hist)
        cost_runs.append(cost_hist)
        kappa_runs.append(kappa_hist)
        mu_runs.append(mu_hist)
        sigma_runs.append(sigma_hist)
        # prob_runs.append(prob_hist)
        f_runs.append(f_mean)
        x_hist.append(x_last)
        y_hist.append(y_last)

    dual_runs = np.array(dual_runs)
    lam_runs = np.array(lam_runs)
    cost_runs = np.array(cost_runs)
    kappa_runs = np.array(kappa_runs)
    mu_runs = np.array(mu_runs)
    sigma_runs = np.array(sigma_runs)
    # prob_runs = np.array(prob_runs)
    f_runs = np.array(f_runs)
    x_hist = np.array(x_hist)
    y_hist = np.array(y_hist)

    # sub_folder = "{}_{}_{}{}-{}.{}.{}".format('vol_str', args.causal, datetime.now().strftime("%h"),
    #                                           datetime.now().strftime("%d"),
    #                                           datetime.now().strftime("%H"),
    #                                           datetime.now().strftime("%M"),
    #                                           datetime.now().strftime("%S"))

    sub_folder = '{}_{}_{:.1f}_{}.{}'.format('vol_str', args.causal, 1/dt,
                                             datetime.now().strftime('%M'), datetime.now().strftime('%S'))

    log_dir = './logs/{}'.format(sub_folder)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save params configuration
    with open('{}/params.txt'.format(log_dir), 'w') as fp:
        fp.write('Params setting \n')
        fp.write('COT or not: {} \n'.format(args.causal))
        fp.write('Vol params: V0 {}, kappa {}, mu {}, sigma {} \n'.format(V_last, kappa, mu, sigma))
        fp.write('batch size {}, seq_len {}, dt {}, n_iter {} \n'.format(small_batch, seq_len, dt, n_iter))
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

    with open('{}/kappa.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(kappa_runs, fp)

    with open('{}/mu.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(mu_runs, fp)

    with open('{}/sigma.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(sigma_runs, fp)

    # with open('{}/prob.pickle'.format(log_dir), 'wb') as fp:
    #     pickle.dump(prob_runs, fp)

    with open('{}/x_hist.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(x_hist, fp)

    with open('{}/y_hist.pickle'.format(log_dir), 'wb') as fp:
        pickle.dump(y_hist, fp)
