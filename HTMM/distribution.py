import torch.nn as nn
import torch as th
import HTMM.th_logprob as thlp
from preprocessing.utils import ConstValues
import torch.distributions as distr

class Categorical(thlp.ProbModule):

    # model distribution of type P(S_L+1 | s_1, ..., s_L). L can be = 1
    def __init__(self, h_size, num_vars, num_labels):
        super(Categorical, self).__init__()
        size = [h_size for i in range(num_vars)] + [num_labels]
        self.p = nn.Parameter(th.empty(size), requires_grad=False)
        self.reset_parameters()

    # def reset_parameters(self):
    #    if self.p.shape[0] == self.p.shape[1]:
    #        self.p.data = th.eye(self.p.shape[0])
    #    else:
    #        alpha = 0.2
    #        dirich = distr.Dirichlet(th.tensor([alpha] * self.p.shape[-1]))
    #        self.p.data = dirich.sample(self.p.shape[:-1]).log()
    #
    #    self.normalise_parameters()
    #    self.reset_posterior()

    # comput p_out given p_hidden
    def forward(self, p_hidden):
        return thlp.sum_over(thlp.mul(p_hidden.unsqueeze(-1), self.p.unsqueeze(0)), 1)

    def set_evidence(self, visible: th.Tensor):
        vis_mask = (visible != ConstValues.NO_ELEMENT)
        return th.gather(self.p, -1, (visible * vis_mask).long().unsqueeze(0).expand(self.p.size(0),-1)).transpose(0, -1) * vis_mask.view(-1, 1)

    def accumulate_posterior(self, posterior, visible):
        if self.training:
            vis_mask = visible != ConstValues.NO_ELEMENT
            # TODO: check if works with n_var > 2
            self.p.grad.index_add_(-1, visible[vis_mask], posterior[vis_mask, :].transpose(0, -1).exp())
