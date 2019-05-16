import numpy as np
import torch
from torch.optim.optimizer import Optimizer
import torch.autograd as autograd
from copy import deepcopy
from functools import reduce


# Adapted from: https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html
class StochasticCubicRegularizedNewton(Optimizer):
    def __init__(self, params, l, rho, epsilon):
        '''
        :param params: model params
        :param l: Lipschitz gradient
        :param rho: Lipschitz hessian
        :param epsilon: final tolerance
        '''
        T_eps = l / np.sqrt(rho * epsilon)
        defaults = dict(l=l,
                        rho=rho,
                        epsilon=epsilon,
                        T_eps=T_eps)
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("SCRN doesn't support per-parameter options "
                             "(parameter groups)")
        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self, params):
        views = [p.grad.view(-1) for p in params]
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def step(self, gradient_for_hess_product):
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        l = group['l']
        rho = group['rho']
        epsilon = group['epsilon']
        T_eps = group['T_eps']

        state = self.state[self._params[0]]
        state.setdefault('n_iter', 0)
        state.setdefault('hit_early_stop', False)

        if state['hit_early_stop']:
            print("Hit the early stop condition already")

        flat_grad = deepcopy(self._gather_flat_grad(self._params))
        flat_grad_for_hess_product = self._gather_flat_grad(gradient_for_hess_product)

        delta, delta_m = self._cubic_subsolver(flat_grad, flat_grad_for_hess_product, group)

        self._add_grad(1, delta)

        if delta_m >= -1./100 * np.sqrt(epsilon**3 / rho):
            delta = self._cubic_finalsolver(flat_grad, flat_grad_for_hess_product, group)
            self._add_grad(1, delta)
            state['hit_early_stop'] = True
        state['n_iter'] += 1

    def _cubic_subsolver(self, flat_grad, flat_grad_for_hess_product, group):
        l = group['l']
        rho = group['rho']
        epsilon = group['epsilon']
        T_eps = group['T_eps']

        # norm_g

        if torch.norm(flat_grad, p=2) >= l**2 / rho:
            self.zero_grad()
            pre_hess_vec_prod = flat_grad_for_hess_product @ flat_grad
            pre_hess_vec_prod.backward()

            hess_vec_prod = self._gather_flat_grad(self._params)

            gBg = flat_grad @ hess_vec_prod

            Rc = - gBg/



