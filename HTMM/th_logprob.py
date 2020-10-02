import torch as th
import torch.nn as nn
import torch.nn.init as INIT


EPS = 10**-44
LOG_EPS = -100  # log(EPS) = LOG_EPS


def mul(a, b):
    return a+b


def div(a, b):
    return a-b


def exp_normalise(self, sum_over_var, get_Z=False):
    Z = th.sum(self, sum_over_var, keepdim=True)
    if get_Z:
        return self / Z.expand_as(self), Z
    else:
        return self / Z.expand_as(self)


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


def sum_over(self, sum_over_var, keepdim=False):
    max = self.max()
    exp_val = th.exp(self - max)
    return th.log(th.sum(exp_val, sum_over_var, keepdim)+EPS) + max


def zeros(*shape):
    return th.full(shape, LOG_EPS)


class ProbModule(nn.Module):

    def __init__(self):
        super(ProbModule, self).__init__()

    def reset_parameters(self):
        for x in self.parameters(recurse=False):
            INIT.uniform_(x)

        self.normalise_parameters()
        self.reset_posterior()

    def normalise_parameters(self):
        for x in self.parameters(recurse=False):
            x.data = normalise(x.data, -1)

    def reset_posterior(self):
        for x in self.parameters(recurse=False):
            x.grad = th.zeros_like(x.data) + EPS

    def m_step(self):
        for n, p in self.named_parameters(recurse=False):
            p.data = exp_normalise(p.grad, -1).log()

        self.reset_posterior()

        for n, m in self.named_children():
            m.m_step()

    def accumulate_posterior(self, *args):
        pass
