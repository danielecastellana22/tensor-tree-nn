import torch.nn as nn
import torch as th
import models.prob.th_logprob as thlp
from preprocessing.utils import ConstValues
import torch.nn.init as INIT
import math


class Categorical(thlp.CategoricalProbModule):

    # model distribution of type P(S_L+1 | s_1, ..., s_L). L can be = 1
    def __init__(self, h_size, num_labels, num_vars=1, alpha=1.):
        super(Categorical, self).__init__()
        self.num_vars = num_vars
        size = [h_size for i in range(num_vars)] + [num_labels]
        self.p = nn.Parameter(th.empty(size), requires_grad=False)

        self.init_parameters(alpha=alpha)
        self.reset_posterior()

    # comput p_out given p_hidden
    def forward(self, *p_hidden):
        assert len(p_hidden) == self.num_vars
        out = self.p.unsqueeze(0)
        for i, h in enumerate(p_hidden):
            out = thlp.sum_over(thlp.mul(h.view(*(list(h.shape) + [1]*(self.num_vars-i))), out), 1)

        return out

    def set_evidence(self, visible: th.Tensor):
        vis_mask = (visible != ConstValues.NO_ELEMENT)
        idx = (visible * vis_mask).long().view([1]*self.num_vars + [-1]).expand(list(self.p.shape[:-1])+[-1])
        return th.gather(self.p, -1, idx).permute([-1] + list(range(self.p.ndim-1))) * vis_mask.view([-1] + [1]*self.num_vars)
        #return th.index_select(self.p, -1, (visible * vis_mask).long()).permute([-1] + list(range(self.p.ndim-1))) * vis_mask.view(-1, 1)

    def accumulate_posterior(self, posterior, visible):
        if self.training:
            vis_mask = visible != ConstValues.NO_ELEMENT
            self.p.grad.index_add_(-1, visible[vis_mask], posterior[vis_mask, :].permute(list(range(1, posterior.ndim))+[0]).exp())


class Normal(thlp.ProbModule):

    def __init__(self, h_size, out_size):
        super(Normal, self).__init__()
        self.h_size = h_size
        self.out_size = out_size

        self.mu = nn.Parameter(th.empty(h_size, out_size), requires_grad=False)
        self.sigma = nn.Parameter(th.empty(h_size, out_size), requires_grad=False)
        self.all_visibles = []
        self.all_posteriors = []
        self.m_step_denom = th.zeros((h_size, 1))

        self.Z = self.out_size * th.tensor(2*math.pi).log()

        self.init_parameters()

    def init_parameters(self):
        INIT.normal_(self.mu, 0, 0.1)
        INIT.constant_(self.sigma, 1)

        self.reset_posterior()

    def reset_posterior(self):
        super(Normal, self).reset_posterior()
        self.all_visibles = []
        self.all_posteriors = []

        self.m_step_denom.fill_(thlp.EPS)

    def forward(self, p_hidden):
        raise NotImplementedError('Cannot sample from normal emission!')

    def set_evidence(self, visible: th.Tensor):
        # visible has shape bs x out_size
        bs = visible.size(0)
        diff = self.mu.unsqueeze(0).expand(bs, -1, -1) - visible.unsqueeze(1).expand(-1, self.h_size, -1)
        loglike = - (diff*diff / self.sigma.unsqueeze(0)).sum(2) - (self.Z + th.log(self.sigma).sum(1))
        loglike = loglike/2
        return loglike

    def accumulate_posterior(self, posterior: th.Tensor, visible_vec: th.Tensor):
        # posterior has shape bs x h_size
        # visible_vec has shape bs x out_size
        posterior_exp = posterior.exp()
        bs = posterior_exp.size(0)
        acc = posterior_exp.unsqueeze(2) * visible_vec.unsqueeze(1)
        self.mu.grad += acc.sum(0)
        self.m_step_denom += posterior_exp.sum(0).unsqueeze(1)

        self.all_posteriors.append(posterior_exp)
        self.all_visibles.append(visible_vec)

        #diag_element = (visible_vec.unsqueeze(1).expand(bs, self.h_size, self.out_size) -
        #                self.mu.unsqueeze(0).expand(bs, self.h_size, self.out_size)).pow(2)
        # has shape bs x h_size x out_size
        #self.sigma.grad += (posterior_exp.unsqueeze(2) * diag_element).sum(0)

    def __m_step__(self):
        self.mu.data = self.mu.grad / self.m_step_denom

        for i in range(len(self.all_visibles)):
            vis_i = self.all_visibles[i]
            post_i = self.all_posteriors[i]
            diag_element = (vis_i.unsqueeze(1).expand(-1, self.h_size, self.out_size) -
                            self.mu.unsqueeze(0).expand(-1, self.h_size, self.out_size)).pow(2)

            self.sigma.grad += (diag_element * post_i.unsqueeze(2)).sum(0)

        self.sigma.data = self.sigma.grad / self.m_step_denom
        my_eps = 10**-6
        self.sigma.data[self.sigma.data<my_eps] = my_eps


