from abc import abstractmethod
from collections.abc import Iterable
import torch as th
import torch.nn as nn
import torch.distributions as distr
import numpy as np


EPS = 10**-44
LOG_EPS = -100  # log(EPS) = LOG_EPS


def mul(a, b):
    return a+b


def div(a, b):
    return a-b


# RETURN LOG
def exp_normalise(self, sum_over_var, get_Z=False):
    Z_log = th.sum(self, sum_over_var, keepdim=True).log()
    if get_Z:
        return self.log() - Z_log.expand_as(self), Z_log
    else:
        return self.log() - Z_log.expand_as(self)


def normalise(self, sum_over_var, get_Z=False):
    # sum_over = [i for i in range(self.ndim) if i not in keep_var]
    Z = sum_over(self, sum_over_var, keepdim=True)
    if get_Z:
        return self - Z.expand_as(self), Z
    else:
        return self - Z.expand_as(self)


def normalise_(self, sum_over_var):
    # sum_over = [i for i in range(self.ndim) if i not in keep_var]
    Z = sum_over(self, sum_over_var, keepdim=True)
    self.data = self - Z.expand_as(self)


# TODO: is there a best way to compute max over multiple dimension?
def sum_over(self, sum_over_var, keepdim=False):
    if isinstance(sum_over_var, Iterable):
        max = self
        for i in sum_over_var:
            max, _ = th.max(max, i, keepdim=True)
    else:
        max, _ = th.max(self, sum_over_var, keepdim=True)

    exp_val = th.exp(self - max)
    if keepdim:
        return th.log(th.sum(exp_val, sum_over_var, keepdim)+EPS) + max
    else:
        #return th.log(th.sum(exp_val, sum_over_var, keepdim) + EPS) + max.squeeze(sum_over_var)
        new_shape = list(np.delete(np.array(max.shape), sum_over_var))
        return th.log(th.sum(exp_val, sum_over_var, keepdim) + EPS) + max.view(*new_shape)


def zeros(*shape):
    return th.full(shape, LOG_EPS)


class ProbModule(nn.Module):

    def __init__(self):
        super(ProbModule, self).__init__()

    def init_parameters(self, **kwargs):
        pass

    def reset_posterior(self):
        for x in self.parameters(recurse=False):
            x.grad = th.zeros_like(x.data) + EPS

    @abstractmethod
    def __m_step__(self):
        raise NotImplementedError('Must be implemented in subclasses!')

    def m_step(self):
        # propagate on other prob modules
        for n, m in self.named_children():
            if isinstance(m, ProbModule):
                m.m_step()

        # do m step on self
        self.__m_step__()
        self.reset_posterior()

    def accumulate_posterior(self, *args):
        pass


# Module which contain only categorical distribution as paramter.
# Morever, each parameter sum to one on the last dimension
class CategoricalProbModule(ProbModule):

    def __init__(self):
        super(CategoricalProbModule, self).__init__()

    def init_parameters(self, alpha=1.):
        for x in self.parameters(recurse=False):
           dirich = distr.Dirichlet(th.tensor([alpha] * x.shape[-1]))
           x.data = dirich.sample(x.shape[:-1]).log()
        #    INIT.uniform_(x)
        #self.normalise_parameters()

    def normalise_parameters(self):
        for x in self.parameters(recurse=False):
            x.data = normalise(x.data, -1)

    def __m_step__(self):
        for n, p in self.named_parameters(recurse=False):
            p.data = exp_normalise(p.grad, -1)

    def accumulate_posterior(self, *args):
        pass
