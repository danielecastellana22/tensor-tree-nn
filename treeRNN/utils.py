import torch.nn as nn
import torch as th
import torch.nn.init as INIT
import numpy as np


class LinearWithTypes(nn.Module):

    def __init__(self, in_size, out_size, type_embs_size):
        super(LinearWithTypes, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.type_emb_size = type_embs_size

        self.W = nn.Parameter(th.Tensor(in_size, out_size))
        self.b = nn.Parameter(th.Tensor(1, out_size))

        if type_embs_size is not None:
            raise NotImplementedError('Using type embs is not allowed yet!')

        self.reset_parameters()

    def reset_parameters(self):
        INIT.xavier_uniform_(self.W)
        INIT.xavier_uniform_(self.b)

    def forward(self, in_el, type_embs=None):

        if type_embs is not None:
            raise NotImplementedError('Using type embs is not allowed yet!')

        bs = in_el.size(0)
        ris = th.matmul(in_el.view(bs, 1, -1), self.W).squeeze(1) + self.b

        return ris


class AugmentedTensor(nn.Module):

    def __init__(self, n_aggr, in_size_list, out_size, pos_stationarity):

        if n_aggr * np.prod(in_size_list) * out_size > 10**9:
            raise ValueError('Too many parameters!')

        super(AugmentedTensor, self).__init__()

        self.n_aggr = n_aggr
        # +1 for the bias
        self.in_size_list = [x+1 for x in in_size_list]
        self.n_input = len(in_size_list)
        self.out_size = out_size
        self.pos_stationarity = pos_stationarity

        if self.pos_stationarity:
            raise NotImplementedError('Full with stationarity not implemented yet')
        else:
            d = self.in_size_list
            d.append(out_size)
            # the n_aggr for batching slow down everything
            self.T_list = nn.ParameterList()
            for i in range(self.n_aggr):
                self.T_list.append(nn.Parameter(th.Tensor(*d)))

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.T_list:
            INIT.xavier_uniform_(p)

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, *in_el_list):
        if self.pos_stationarity:
            raise NotImplementedError('Full with stationarity not implemented yet')

        bs = in_el_list[0].size(0)
        n_aggr = self.n_aggr

        ris_list = []
        h = th.cat((in_el_list[0].view(bs, n_aggr, -1), th.ones((bs, n_aggr, 1), device=in_el_list[0].device)), dim=2)
        for j in range(n_aggr):
            ris_list.append(th.matmul(h[:, j, :], self.T_list[j].view(self.in_size_list[0], -1)))

        for i in range(1, self.n_input):
            in_el = in_el_list[i] # has shape (bs x n_aggr x h)
            dim_i = self.in_size_list[i]
            h = th.cat((in_el, th.ones((bs, n_aggr, 1), device=in_el.device)), dim=2)
            for j in range(n_aggr):
                ris_list[j] = th.bmm(h[:, j:j+1, :], ris_list[j].view(bs,  dim_i, -1))

        return th.cat(ris_list, dim=1)