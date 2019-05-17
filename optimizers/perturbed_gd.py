import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from copy import deepcopy
from functools import reduce


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

        self._is_done = False
        self.n_iter = 0
        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self.t_noise = - t_thresh - 1
        self.curr_x_tilde = None

    def _gather_flat_grad(self):
        views = [p.grad.view(-1) for p in self._params]
        return torch.cat(views, 0)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def _update_grad(self, const, tensor):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.zero_()
            p.data.add_(tensor[offset:offset + numel].view_as(p.data))
            p.data.add_(const)
            offset += numel
        assert offset == self._numel()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        assert len(self.param_groups) == 1
        group = self.param_groups[0]

        chi = group['chi']
        eta = group['eta']
        r = group['r']
        g_thresh = group['g_thresh']
        f_thresh = group['f_thresh']
        t_thresh = group['t_thresh']
        d = group['d']


        flat_grad = self._gather_flat_grad()

        if torch.norm(flat_grad, p=2) <= g_thresh and self.n_iter - self.t_noise > t_thresh:
            self.curr_x_tilde = flat_grad
            self.t_noise = self.n_iter

            # y = np.random.multivariate_normal(np.zeros(d), np.identity(d))
            xi = torch.tensor(np.random.normal(loc=0,scale=1, size=len(flat_grad))).float()
            xi.div_(torch.norm(xi, p=2))
            u = np.random.uniform(0, r)
            xi.mul_(u**(1./d))
            self._update_grad(xi, self.curr_x_tilde)
        self._add_grad(-eta, flat_grad)
        self.n_iter += 1

        return loss
