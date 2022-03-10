import torch

def modi_cost(x, y, h, M, lam, f, cost):
    '''
    :param x: a tensor of shape [batch_size, time steps, features]
    :param y: a tensor of shape [batch_size, time steps, features]
    :param h: a tensor of shape [batch size, time steps, J]
    :param M: a tensor of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :return: L1 cost matrix plus h, M modification:
    a matrix of size [batch_size, batch_size] where
    c_hM_{ij} = c_hM(x^i, y^j) = L2_cost + \sum_{t=1}^{T-1}h_t\Delta_{t+1}M
    ====> NOTE: T-1 here, T = # of time steps
    '''
    # compute sum_{t=1}^{T-1} h[t]*(M[t+1]-M[t])
    DeltaM = M[:, 1:, :] - M[:, :-1, :]
    # ht = h[:, :-1, :]
    time_steps = h.shape[1]
    sum_over_j = torch.sum(h[:, None, :, :] * DeltaM[None, :, :, :], -1)
    C_hM = torch.sum(sum_over_j, -1) / time_steps

    return f(y).repeat(x.shape[0], 1) - lam*cost(x, y) + C_hM


def compute_sinkhorn(x, y, h, M, lam, f, cost, eps=0.01, niter=20):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    epsilon = lam*eps + 1e-5
    n = x.shape[0]
    # The Sinkhorn algorithm takes as input three variables :
    C = modi_cost(x, y, h, M, lam, f, cost) # shape: [batch_size, batch_size]

    # Add robustness
    # C = torch.minimum(C, torch.tensor([5000]))
    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    mu = 1. / n * torch.ones(n, device=x.device)
    nu = 1. / n * torch.ones(n, device=x.device)

    # Parameters of the Sinkhorn algorithm.
    # rho = 1  # (.5) **2          # unbalanced transport
    # tau = -.8  # nesterov-like acceleration
    # lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-4)  # stopping criterion

    # Elementary operations .....................................................................
    # def ave(u, u1):
    #     "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        # return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.logsumexp(A, dim=-1, keepdim=True)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).item():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost, pi

def martingale_regularization(M, reg=100.0):
    '''
    Compute the regularization for the martingale condition (i.e. p_M).
    :param M: a tensor of shape (batch_size, sequence length), the output of an RNN applied to X
    :param reg_lam: scale parameter for first term in pM
    :return: A rank 0 tensors (i.e. scalers)
    This tensor represents the martingale penalization term denoted $p_M$
    '''
    m, t, j = M.shape
    # m = torch.tensor(m).type(torch.FloatTensor)
    # t = torch.tensor(m).type(torch.FloatTensor)
    # compute delta M matrix N
    N = M[:, 1:, :] - M[:, :-1, :]
    N_std = N / (torch.std(M, (0, 1)) + 1e-06)

    # Compute \sum_i^m(\delta M)
    sum_m_std = torch.sum(N_std, 0) / m
    # Compute martingale penalty: P_M1 =  \sum_i^T(|\sum_i^m(\delta M)|) * scaling_coef
    sum_across_paths = torch.sum(torch.abs(sum_m_std)) / t
    # the total pM term
    pm = reg * sum_across_paths
    return pm
