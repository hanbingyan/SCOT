import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
from nets import basis_net
from sp_config import *
from utils import martingale_regularization

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

def c(x, y, p=2):
    return (x - y).pow(p).sum(axis=(1, 2))/x.shape[0]/x.shape[1]/x.shape[2]


def beta(x, M, test_H, f, lam):

    martingale_diff = M[:, 1:, :] - M[:, :-1, :]
    y = (x + torch.rand_like(x) * 1e-3).clone().requires_grad_()

    t = 0
    b_old = 10
    h = test_H(y).detach()[:, :-1, :]
    b = f(y).sum() + torch.sum(h*martingale_diff) - lam*c(x, y).sum()
    # b = f(y).sum() - lam * cost(x, y).sum()
    while t<20 and torch.abs((b - b_old)/b_old) > 1e-3:

        b_old = b.item()
        t += 1
        b.backward(retain_graph=True)
        y = torch.clamp(y + 20/(t+1)*y.grad/y.grad.abs().sum(), min=0.0, max=3.0).clone().detach().requires_grad_()
        h = test_H(y).detach()[:, :-1, :]

        b = f(y).sum() + torch.sum(h * martingale_diff) - lam * c(x, y).sum()
        # b = f(y).sum() - lam * cost(x, y).sum()
    # print(t)
    return y

def train(SP_sampler, f, idx, causal):

    test_H = basis_net(in_size=in_feature, hid_size=4, out_size=out_feature, init=False, req_grad=causal).to(device)
    test_M = basis_net(in_size=in_feature, hid_size=4, out_size=out_feature, init=True).to(device)
    var_hist = torch.zeros(n_iter, device='cpu')
    lam_hist = torch.zeros(n_iter, device='cpu')
    cost_hist = torch.zeros(n_iter, device='cpu')
    f_mean = torch.zeros(n_iter, device='cpu')
    lam = torch.tensor([lam_init], requires_grad=True, device=device)

    optimMH = optim.Adam(list(test_M.parameters()) + list(test_H.parameters()), lr = 1e-4)

    velocity = 0.0

    for iter in range(n_iter):

        ## Inner minimization over lambda, h, g
        x_tar = SP_sampler(small_batch, end_idx=idx)
        input_x = x_tar[:, :-1, :]
        tar = np.log(x_tar[:, -1, 0])
        x = torch.tensor(input_x, dtype=torch.float, device=device)

        M = test_M(x)
        y = beta(x, M, test_H, f, lam.detach())

        M = test_M(x)
        martingale_diff = M[:, 1:, :] - M[:, :-1, :]
        h = test_H(y.detach())[:, :-1, :]
        b = f(y.detach()).sum() + torch.sum(h * martingale_diff) - lam * c(x, y.detach()).sum()
        # b = f(y).sum() - lam * cost(x, y).sum()
        dual = lam * radius + b / small_batch
        obj = dual + martingale_regularization(M, reg=10.0)

        test_H.zero_grad()
        test_M.zero_grad()
        lam.grad = None
        obj.backward()
        D_lam = lam.grad
        optimMH.step()

        velocity = 0.9 * velocity - 1 / (iter + 100) * D_lam
        lam = torch.clamp(lam + velocity, min=0.0).clone().detach().requires_grad_()

        with torch.no_grad():
            for param in test_M.parameters():
                param.clamp_(-0.2, 0.2)


        var_hist[iter] = dual.item()
        lam_hist[iter] = lam.item()
        cost_hist[iter] = torch.sum(c(x,y))
        f_mean[iter] = f(y).mean().item()
        if iter % 50 == 0:
            # print(bcolors.GREEN, 'Det f', f(y).detach(), bcolors.ENDC)
            # print('target', tar)
            print('iter', iter, 'dual', var_hist[iter].item(), 'lam', lam.item(),
                  'f mean', f_mean[iter], 'real mean', tar.mean(),
                  'f > real', np.sum(f(y).detach().cpu().numpy() > tar.reshape(-1)),
                  'cost', torch.sum(c(x,y)).item())
            # for name, param in gen_Y.named_parameters():
            #     print(name, param)

    sub_folder = "{}_{}_{}{}-{}.{}.{}".format('spx_dety', idx, datetime.now().strftime("%h"),
                                           datetime.now().strftime("%d"),
                                           datetime.now().strftime("%H"),
                                           datetime.now().strftime("%M"),
                                           datetime.now().strftime("%S"))
    log_dir = "./logs/{}".format(sub_folder)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save params configuration
    with open('{}/params.txt'.format(log_dir), 'w') as fp:
        fp.write('Params setting \n')
        fp.write('idx {} \n'.format(idx))
        fp.write('batch size {}, seq_len {}, n_iter {} \n'.format(small_batch, seq_len, n_iter))
        fp.write('Radius: {} \n'.format(radius))
        fp.write('Lambda Init: {} \n'.format(lam_init))

    if causal:
        with open('{}/dety_cot_var.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(var_hist, fp)

        with open('{}/dety_cot_f.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(f_mean, fp)

        with open('{}/dety_cot_cost.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(cost_hist, fp)
    else:
        with open('{}/dety_ot_var.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(var_hist, fp)

        with open('{}/dety_ot_f.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(f_mean, fp)

        with open('{}/dety_ot_cost.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(cost_hist, fp)


    ### out-of-sample test ###

    # end_idx = max_len - 500
    # sim_x_tar = pool.map(SP_x_tar, [k for k in range(100)])
    x_tar = SP_sampler(small_batch, end_idx=idx, rand=False)
    input_x = x_tar[:, :-1, :]
    tar = np.log(x_tar[:, -1, 0])
    in_x = torch.tensor(input_x, dtype=torch.float, device=device)

    M = test_M(in_x)
    y = beta(in_x, M, test_H, f, lam.detach())
    predicted = f(y).cpu()

    print('Target', tar)
    print('Robust predicted', predicted.detach())
    nonrobust = f(in_x).cpu()
    print('Non robust predicted', nonrobust.detach())

    print('Predicted mean', predicted.detach().numpy().mean(), 'real mean', tar.mean(),
          'f > real', np.sum(predicted.detach().numpy() > tar.reshape(-1)))
    print('Mean Absolute Error', np.sum(np.abs(predicted.detach().numpy() - tar.reshape(-1))))
    print('Correct direction', np.sum(np.multiply(predicted.detach().numpy(), tar.reshape(-1)) > 0))

    print('Nonrobust mean', nonrobust.detach().numpy().mean(),
          'Nonrobust > real', np.sum(nonrobust.detach().numpy() > tar.reshape(-1)))
    print('Nonrobust MAE', np.sum(np.abs(nonrobust.detach().numpy() - tar.reshape(-1))))
    print('Correct direction', np.sum(np.multiply(nonrobust.detach().numpy(), tar.reshape(-1)) > 0))

    if causal:
        with open('{}/dety_target.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(tar, fp)

        with open('{}/dety_robust.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(predicted.detach().numpy(), fp)

        with open('{}/dety_nonrobust.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(nonrobust.detach().numpy(), fp)
    else:
        with open('{}/dety_ot_target.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(tar, fp)

        with open('{}/dety_ot_robust.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(predicted.detach().numpy(), fp)

        with open('{}/dety_ot_nonrobust.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(nonrobust.detach().numpy(), fp)

    plt.plot(var_hist.numpy())
    plt.show()

    plt.plot(lam_hist.numpy())
    plt.show()

    return var_hist
