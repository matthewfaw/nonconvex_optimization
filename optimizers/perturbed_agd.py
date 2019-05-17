import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from functools import reduce
from copy import deepcopy

import pprint
pp = pprint.PrettyPrinter(indent=4)

class PerturbedAGD(Optimizer):
    def __init__(self,
                 params,
                 eta,
                 theta,
                 gamma,
                 s,
                 r,
                 T,
                 epsilon,
                 add_noise=True,
                 neg_curv_explore=True):
        defaults = dict(eta=eta,
                        theta=theta,
                        gamma=gamma,
                        s=s,
                        r=r,
                        T=T,
                        epsilon=epsilon,
                        add_noise=add_noise,
                        neg_curv_explore=neg_curv_explore)
        print("Using defaults")
        pp.pprint(defaults)
        super().__init__(params, defaults)

        self._is_done = False
        self.n_iter = 0
        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self.curr_x_tilde = None
        self.t_noise = - T - 1
        self.curr_v = None

    def _gather_flat_grad(self):
        views = [p.grad.view(-1) for p in self._params]
        return torch.cat(views, 0)

    def _gather_flat_tensor(self):
        views = [p.view(-1) for p in self._params]
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

    def _update_params(self, const, tensor):
        offset = 0
        for p_id in range(len(self._params)):
            p = self._params[p_id]
            tens_p = tensor[p_id]

            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.zero_()
            p.data.add_(tens_p)
            p.data.add_(const)
            offset += numel
        assert offset == self._numel()

    def _negative_curvature_exploration(self, x_t, v_t, s, closure):
        norm_vt = torch.norm(v_t, p=2)
        if norm_vt >= s:
            x_t_plus = x_t
        else:
            delta = s/norm_vt * v_t

            cand_1 = x_t + delta
            self._update_params(0, cand_1)
            loss_1 = closure()
            cand_2 = x_t - delta
            self._update_params(0, cand_2)
            loss_2 = closure()

            if loss_1 < loss_2:
                return cand_1, torch.zeros_like(cand_1)
            else:
                return cand_2, torch.zeros_like(cand_2)

    def step(self, closure):
        loss_at_start = closure()

        assert len(self.param_groups) == 1
        group = self.param_groups[0]

        eta = group['eta']
        theta = group['theta']
        gamma = group['gamma']
        s = group['s']
        r = group['r']
        T = group['T']
        epsilon = group['epsilon']
        add_noise = group['add_noise']
        neg_curv_explore = group['neg_curv_explore']

        flat_grad = deepcopy(self._gather_flat_grad())
        d = len(flat_grad)
        if self.curr_v is None:
            self.curr_v = torch.zeros_like(flat_grad)

        if add_noise and torch.norm(flat_grad, p=2) <= epsilon and self.n_iter - self.t_noise > T:
            self.curr_x_tilde = deepcopy(self._params)
            self.t_noise = self.n_iter

            # y = np.random.multivariate_normal(np.zeros(d), np.identity(d))
            xi = torch.tensor(np.random.normal(loc=0,scale=1, size=d)).float()
            xi.div_(torch.norm(xi, p=2))
            u = np.random.uniform(0, r)
            xi.mul_(u**(1./d))
            self._update_params(xi, self.curr_x_tilde)

        x_t_flat = self._gather_flat_tensor()
        curr_y = self.curr_v.mul(1 - theta)
        curr_y.add_(x_t_flat)

        self._update_params(0, curr_y)
        loss_at_yt = closure()
        grad_at_y1 = self._gather_flat_tensor()
        self._add_grad(-eta, grad_at_y1)

        x_t_plus = self._gather_flat_tensor()
        self.curr_v = x_t_plus - x_t_flat

        if neg_curv_explore and loss_at_start <= loss_at_yt + (grad_at_y1 @ (x_t_flat - curr_y)) - gamma/2 * torch.norm(x_t_flat - curr_y, p=2)**2:
            print("Exploring neg curv")
            x_t_plus, self.curr_v = self._negative_curvature_exploration(x_t_flat, self.curr_v, s, closure)
            self._update_params(0, x_t_plus)

        self.n_iter += 1

        return loss_at_start
