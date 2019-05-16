import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from copy import deepcopy


class PerturbedGD(Optimizer):
    def __init__(self, params, l, rho, epsilon, c, delta, delta_f, d):
        chi = 3 * max(np.log(d * l * delta_f / (c * (epsilon**2) * delta)), 4)
        eta = c / l
        r = np.sqrt(c) / chi**2 * epsilon/l
        g_thresh = np.sqrt(c) / chi**2 * epsilon
        f_thresh = c/chi**3 * np.sqrt(epsilon**3 / rho)
        t_thresh = chi / c**2 * l/np.sqrt(rho * epsilon)
        defaults = dict(chi=chi,
                        eta=eta,
                        r=r,
                        g_thresh=g_thresh,
                        f_thresh=f_thresh,
                        t_thresh=t_thresh,
                        d=d)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            chi = group['chi']
            eta = group['eta']
            r = group['r']
            g_thresh = group['g_thresh']
            f_thresh = group['f_thresh']
            t_thresh = group['t_thresh']
            d = group['d']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                param_state = self.state[p]
                if 't_noise' not in param_state:
                    param_state['t_noise'] = deepcopy(-t_thresh - 1)
                if 'T' not in param_state:
                    param_state['T'] = torch.tensor(0)

                t_noise = param_state['t_noise']
                T = param_state['T']

                if torch.le(d_p, g_thresh).all() and T - t_noise > t_thresh:
                    print("Adding noise")
                    if 'curr_x_tilde' not in param_state:
                        param_state['curr_x_tilde'] = torch.tensor(0)
                    curr_x_tilde = param_state['curr_x_tilde']
                    curr_x_tilde.data = deepcopy(p.data)
                    t_noise = deepcopy(T)

                    y = np.random.multivariate_normal(np.zeros(d), np.identity(d))
                    y_norm = y/np.linalg.norm(y, ord=2)
                    u = np.random.uniform(0, r)
                    xi = u**(1./d)*y_norm

                    p.data = curr_x_tilde + xi
                p.data.add_(-eta, d_p)
                T.add_(1)

        return loss
