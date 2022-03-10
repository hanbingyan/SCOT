import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
from nets import basis_net
from config import *
from utils import *

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

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
    return c/x.shape[2]/x.shape[1]/x.shape[0]


def train(SP_sampler, gen_Y, f, idx, causal):

    test_H = basis_net(in_size=in_feature, hid_size=4, out_size=out_feature, init=False, req_grad=causal).to(device)
    test_M = basis_net(in_size=in_feature, hid_size=4, out_size=out_feature, init=False, req_grad=causal).to(device)
    var_hist = torch.zeros(n_iter, device=device)
    lam_hist = torch.zeros(n_iter, device=device)
    cost_hist = torch.zeros(n_iter, device=device)
    f_mean = torch.zeros(n_iter, device=device)
    lam = torch.tensor([lam_init], requires_grad=True, device=device)

    optimMH = optim.Adam(list(test_M.parameters()) + list(test_H.parameters()), lr = 5e-4)
    optimY = optim.Adam(list(gen_Y.parameters()), lr = 5e-6)
    velocity = 0.0

    for iter in range(n_iter):

        ## Inner minimization over lambda, h, g
        # sample from reference distribution, shape [n_batch, seq+1, n_feature]
        # [:, -1, 0] is the target close price
        # sim_x_tar = pool.map(x_tar_helper, [k for k in range(small_batch)])
        # x_tar = np.array(sim_x_tar)
        x_tar = SP_sampler(small_batch, end_idx=idx)
        input_x = x_tar[:, :-1, :]
        x = torch.tensor(input_x, dtype=torch.float, device=device)

        g = test_M(x)
        z = torch.randn_like(x)*1e-2
        y = gen_Y(x+z).detach()
        # y = gen_Y(x+z).detach()

        h = test_H(y)[:, :-1, :]
        wass_sam, pi = compute_sinkhorn(x, y, h, g, lam, f, c)
        in_loss = lam * radius + wass_sam + martingale_regularization(g)
        test_H.zero_grad()
        test_M.zero_grad()
        lam.grad = None
        in_loss.backward()
        D_lam = lam.grad
        optimMH.step()
        velocity = 0.9*velocity - 1/(0.1*iter + iter_const)*D_lam
        lam = torch.clamp(lam + velocity, min=0.0).clone().detach().requires_grad_()

        with torch.no_grad():
            for param in test_M.parameters():
                # param.add_(torch.randn(param.size(), device=device)/50)
                param.clamp_(-0.5, 0.5)
                # param.clamp_(-0.2, 0.2)
            for param in test_H.parameters():
                # param.add_(torch.randn(param.size(), device=device)/50)
                param.clamp_(-1.0, 1.0)

        #### Maximization over Generator ######
        # sample from reference distribution
        # sim_x_tar = pool.map(x_tar_helper, [k for k in range(small_batch)])
        # x_tar = np.array(sim_x_tar)
        x_tar = SP_sampler(small_batch, end_idx=idx)
        input_x = x_tar[:, :-1, :]
        tar = np.log(x_tar[:, -1, 0])
        x = torch.tensor(input_x, dtype=torch.float, device=device)

        g = test_M(x).detach()
        z = torch.randn_like(x)*1e-2
        y = gen_Y(x+z)
        # y= gen_Y(x+z)

        h = test_H(y)[:, :-1, :].detach()
        out_wass, out_pi = compute_sinkhorn(x, y, h, g, lam.detach(), f, c)
        out_loss = - out_wass
        gen_Y.zero_grad()
        out_loss.backward()
        optimY.step()

        # with torch.no_grad():
        #     for param in gen_Y.parameters():
        #         param.add_(torch.randn(param.size(), device=device)/50)
                # param.clamp_(-0.5, 0.5)

        var_hist[iter] = -out_loss.item() + lam.detach()*radius
        lam_hist[iter] = lam.item()
        cost_hist[iter] = torch.sum(c(x,y)*out_pi) - 0.01*torch.sum(out_pi*torch.log(out_pi))
        # f_mean[iter] = f(x)[:, -1, 0].mean().item()
        f_mean[iter] = f(y).mean().item()
        if iter % 50 == 0:
            # print(bcolors.RED, 'Rand f', f(y).detach(), bcolors.ENDC)
            # print('target', tar)
            print('iter', iter, 'dual', var_hist[iter].item(), 'lam', lam.item(),
                  'f mean', f_mean[iter], 'real mean', tar.mean(),
                  'f > real', np.sum(f(y).detach().numpy() > tar.reshape(-1)),
                  'cost', torch.sum(c(x, y)*out_pi).item(),
                  'entropy', -entropy_const*torch.sum(out_pi*torch.log(out_pi)).item())

            # print('H sum', h.abs().sum())
            # print('G sum', g.abs().sum())

    sub_folder = "{}_{}_{}_{}{}-{}.{}.{}".format('spx_randy', causal, idx, datetime.now().strftime("%h"),
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
        fp.write('COT or not: {} \n'.format(causal))
        fp.write('idx {} \n'.format(idx))
        fp.write('batch size {}, seq_len {}, n_iter {} \n'.format(small_batch, seq_len, n_iter))
        fp.write('Radius: {} \n'.format(radius))
        fp.write('Lambda Init: {} \n'.format(lam_init))

    if causal:
        with open('{}/randy_cot_var.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(var_hist, fp)

        with open('{}/randy_cot_f.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(f_mean, fp)

        with open('{}/randy_cot_cost.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(cost_hist, fp)
    else:
        with open('{}/randy_ot_var.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(var_hist, fp)

        with open('{}/randy_ot_f.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(f_mean, fp)

        with open('{}/randy_ot_cost.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(cost_hist, fp)


    ### out-of-sample test ###

    # end_idx = max_len - 500
    # sim_x_tar = pool.map(SP_x_tar, [k for k in range(100)])
    x_tar = SP_sampler(small_batch, end_idx=idx, rand=False)
    input_x = x_tar[:, :-1, :]
    tar = np.log(x_tar[:, -1, 0])
    in_x = torch.tensor(input_x, dtype=torch.float, device=device)

    z = torch.randn_like(in_x) * 1e-3
    y = gen_Y(in_x + z)
    predicted = f(y)

    print('Target', tar)
    print('Robust predicted', predicted.detach())
    nonrobust = f(in_x)
    print('Non robust predicted', nonrobust.detach())

    print('Predicted mean', predicted.detach().numpy().mean(), 'real mean', tar.mean(),
          'f > real', np.sum(predicted.detach().numpy() > tar.reshape(-1)))
    print('Mean Absolute Error', np.sum(np.abs(predicted.detach().numpy() - tar.reshape(-1))))
    print('Correct direction', np.sum(np.multiply(predicted.detach().numpy(), tar.reshape(-1)) > 0))

    print('Nonrobust mean', nonrobust.detach().numpy().mean(), 'Nonrobust > real', np.sum(nonrobust.detach().numpy() > tar.reshape(-1)))
    print('Nonrobust MAE', np.sum(np.abs(nonrobust.detach().numpy() - tar.reshape(-1))))
    print('Correct direction', np.sum(np.multiply(nonrobust.detach().numpy(), tar.reshape(-1)) > 0))

    if causal:
        with open('{}/target.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(tar, fp)

        with open('{}/robust.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(predicted.detach().numpy(), fp)

        with open('{}/nonrobust.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(nonrobust.detach().numpy(), fp)
    else:
        with open('{}/ot_target.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(tar, fp)

        with open('{}/ot_robust.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(predicted.detach().numpy(), fp)

        with open('{}/ot_nonrobust.pickle'.format(log_dir), 'wb') as fp:
            pickle.dump(nonrobust.detach().numpy(), fp)



    plt.plot(var_hist.numpy())
    plt.show()

    plt.plot(lam_hist.numpy())
    plt.show()

    return var_hist
