import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from copy import deepcopy
from functools import reduce
import pprint
pp = pprint.PrettyPrinter(indent=4)


# Adapted from: https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html
class StochasticCubicRegularizedNewton(Optimizer):
    def __init__(self, params, l, rho, epsilon, c_prime):
        '''
        :param params: model params
        :param l: Lipschitz gradient
        :param rho: Lipschitz hessian
        :param epsilon: final tolerance
        :param c_prime:
        '''
        T_eps = int(l / np.sqrt(rho * epsilon))
        defaults = dict(l=l,
                        rho=rho,
                        epsilon=epsilon,
                        T_eps=T_eps,
                        c_prime=c_prime)
        print("Defaults:")
        pp.pprint(defaults)
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("SCRN doesn't support per-parameter options "
                             "(parameter groups)")
        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self._is_done = False
        self.n_iter = 0

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

    def step(self,flat_grad_for_hess_product):
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
            return

        flat_grad = deepcopy(self._gather_flat_grad(self._params))

        delta, delta_m = self._cubic_subsolver(flat_grad, flat_grad_for_hess_product, group)

        self._add_grad(1, delta)

        if delta_m >= -1./100 * np.sqrt(epsilon**3 / rho):
            delta = self._cubic_finalsolver(flat_grad, flat_grad_for_hess_product, group)
            self._add_grad(1, delta)
            state['hit_early_stop'] = True
            self._is_done = True
            print("Hit early stopping condition!")
        state['n_iter'] += 1

    def _cubic_subsolver(self, flat_grad, flat_grad_for_hess_product, group):
        l = group['l']
        rho = group['rho']
        epsilon = group['epsilon']
        T_eps = group['T_eps']
        c_prime = group['c_prime']

        norm_g = torch.norm(flat_grad, p=2)

        if norm_g >= l**2 / rho:
            hess_vec_prod = self._get_hess_vec_prod(flat_grad_for_hess_product, flat_grad)

            gBg = flat_grad @ hess_vec_prod

            Rc = - gBg/(rho * norm_g**2) + torch.sqrt((gBg/(rho * norm_g**2)**2 + 2*norm_g/rho))
            delta = -Rc/norm_g * flat_grad

        else:
            delta = torch.zeros_like(flat_grad)
            sigma = c_prime * np.sqrt(epsilon * rho)/l
            eta = 1./(20*l)

            # xi = torch.tensor(np.random.multivariate_normal(mean=np.zeros(len(flat_grad)),
            #                                                 cov=np.diag(np.ones(len(flat_grad)))))
            xi = torch.tensor(np.random.normal(loc=0, scale=1, size=len(flat_grad))).float()
            norm_xi = torch.norm(xi, p=2)
            xi.mul_(1./norm_xi)

            g_tilde = flat_grad.add(sigma, xi)
            for t in range(T_eps):
                hess_del_prod = self._get_hess_vec_prod(flat_grad_for_hess_product, delta)

                delta.add_(-eta, g_tilde + hess_del_prod + rho/2 * torch.norm(delta, p=2) * delta)

        hess_del_prod_2 = self._get_hess_vec_prod(flat_grad_for_hess_product, delta)
        delta_m = flat_grad @ delta + 0.5 * delta @ hess_del_prod_2 + rho/6 * torch.norm(delta,p=2)**3

        return delta, delta_m

    def _cubic_finalsolver(self, flat_grad, flat_grad_for_hess_product, group):
        l = group['l']
        rho = group['rho']
        epsilon = group['epsilon']
        T_eps = group['T_eps']
        c_prime = group['c_prime']

        delta = torch.zeros_like(flat_grad)
        g_m = deepcopy(flat_grad)
        eta = 1./(20*l)

        while torch.norm(g_m) > epsilon/2:
            delta.add_(-eta, g_m)

            hess_del_prod = self._get_hess_vec_prod(flat_grad_for_hess_product, delta)

            g_m = flat_grad + hess_del_prod + rho/2 * torch.norm(delta,p=2)*delta

        return delta

    def _get_hess_vec_prod(self, flat_grad_for_hess_prod, v):
        self.zero_grad()
        if flat_grad_for_hess_prod.grad is not None:
            flat_grad_for_hess_prod.detach_()
            flat_grad_for_hess_prod.zero_()
        pre_hess_vec_prod = flat_grad_for_hess_prod @ v
        pre_hess_vec_prod.backward(retain_graph=True)

        return deepcopy(self._gather_flat_grad(self._params))



