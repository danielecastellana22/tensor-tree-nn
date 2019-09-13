import torch.nn as nn
import torch as th

#TODO: implement all the others aggregator from cell_old


class BaseAggregator(nn.Module):

    # n_aggr allows to speed up the computation computing more aggregation in parallel. USEFUL FOR LSTM
    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr):
        super(BaseAggregator, self).__init__()

        self.n_aggr = n_aggr
        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity

    def forward(self, neighbour_h, nodes):
        pass


# The Full aggregator is available only when maxOutput Degree is 2
# h = A*h1*h2 + U1*h1 + U2*h2 + b
class BinaryFullTensor(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        if max_output_degree > 2:
            raise ValueError('Full cel type can be use only with a maximum output degree of 2')

        super(BinaryFullTensor, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        if self.pos_stationarity:
            #TODO: does not work. Output must be 3*h_size
            raise NotImplementedError('Full with stationarity not implemented yet')
        else:
            # define parameters for the computation of iou
            self.A = nn.Parameter(th.randn(self.h_size, self.h_size, self.n_aggr*self.h_size))
            self.U1 = nn.Parameter(th.randn(self.h_size, self.n_aggr*self.h_size))
            self.U2 = nn.Parameter(th.randn(self.h_size, self.n_aggr*self.h_size))
            self.b = nn.Parameter(th.randn(self.n_aggr*self.h_size))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, nodes):
        if self.pos_stationarity:
            raise NotImplementedError('Full with stationarity not implemented yet')
        else:
            A = self.A
            U1 = self.U1
            U2 = self.U2
            b = self.b
        h1 = neighbour_h[:, 0, :].view(neighbour_h.size(0), -1)
        h2 = neighbour_h[:, 1, :].view(neighbour_h.size(0), -1)
        return th.einsum('ijk,ni,nj->nk', A, h1, h2) + th.matmul(h1, U1) + th.matmul(h2, U2) + b


# h = U1*h1 + U2*h2 + ... + Un*hn
class SumChild(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        super(SumChild, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        if not self.pos_stationarity:
            #NARY aggregation
            # define parameters for the computation of iou
            self.U = nn.Parameter(th.randn(self.max_output_degree * self.h_size, self.n_aggr*self.h_size))
            self.b = nn.Parameter(th.randn(self.n_aggr*self.h_size))
        else:
            #SUMCHILD aggregation
            self.U = nn.Parameter(th.randn(self.h_size, self.n_aggr * self.h_size))
            self.b = nn.Parameter(th.randn(self.n_aggr * self.h_size))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, nodes):
        if not self.pos_stationarity:
            return th.addmm(self.b, neighbour_h.view(neighbour_h.size(0), -1), self.U)
        else:
            return th.addmm(self.b, th.sum(neighbour_h, 1), self.U)

