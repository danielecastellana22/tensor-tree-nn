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

    # input is nieghbour_h has shape batch_size x n_neighbours x h_size
    # output has shape batch_size x (n_aggr * h_size)
    def forward(self, neighbour_h, nodes):
        pass


class FullTensor(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        if h_size**max_output_degree > 10**10:
            raise ValueError('Too many parameters!')

        super(FullTensor, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        if self.pos_stationarity:
            #TODO: does not work. Output must be 3*h_size
            raise NotImplementedError('Full with stationarity not implemented yet')
        else:
            # +1 for the bias
            d = [self.h_size+1 for i in range(max_output_degree)]
            d.insert(1, self.n_aggr*self.h_size)
            self.A = nn.Parameter(th.randn(*d))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, nodes):
        if self.pos_stationarity:
            raise NotImplementedError('Full with stationarity not implemented yet')
        else:
            A = self.A

        ris = None
        n_batch = neighbour_h.size(0)
        for i in range(0, self.max_output_degree):
            if ris is None:
                # add bias. h has size N_BATCH x H_SIZE+1
                h = th.cat((neighbour_h[:, i, :].view(n_batch, -1), th.ones((neighbour_h.size(0), 1), device=neighbour_h.device)), dim=1)
                ris = th.matmul(h, A.view((self.h_size+1, -1)))
            else:
                # add bias. h has size N_BATCH x H_SIZE+1 x 1
                h = th.cat((neighbour_h[:, i, :].view(n_batch, -1, 1), th.ones((neighbour_h.size(0), 1, 1), device=neighbour_h.device)), dim=1)
                ris = th.bmm(ris.view((n_batch, -1, self.h_size+1)), h)

        return ris.squeeze(dim=2)


#TODO: use a LINEAR MDOULE!!!!
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


# h = U3*r3, where r3 = G*r1*r2, r1 = U1*h1 and r2 = U2*h2
class Hosvd(BaseAggregator):

    def __init__(self, h_size, max_output_degree, pos_stationarity, n_aggr, **kwargs):
        super(Hosvd, self).__init__(h_size, max_output_degree, pos_stationarity, n_aggr)

        rank = kwargs['rank']
        self.rank = rank

        # core tensor is a fulltensor aggregator wiht r^d size
        self.G_list = nn.ModuleList()
        for i in range(n_aggr):
            self.G_list.append(FullTensor(rank, max_output_degree, pos_stationarity, n_aggr=1))

        if self.pos_stationarity:
            raise NotImplementedError("pos stationariy is not implemented yet!")
        else:
            # mode matrices
            self.U_list = nn.ModuleList()
            for i in range(max_output_degree):
                self.U_list.append(nn.Linear(h_size, n_aggr*rank, bias=True))

        # output matrices
        self.U_output_list = nn.ModuleList()
        for i in range(n_aggr):
            self.U_output_list.append(nn.Linear(rank, h_size))

    # neighbour_states has shape batch_size x n_neighbours x insize
    def forward(self, neighbour_h, nodes):
        if self.pos_stationarity:
            raise NotImplementedError("pos stationariy is not implemented yet!")
        else:
            # obtain a tensor n_batch x n_neighbours x rank x n_aggr to pass to the full aggregator
            neighbour_r = th.zeros((neighbour_h.size(0), self.max_output_degree, self.rank, self.n_aggr), device=neighbour_h.device)
            for i in range(self.max_output_degree):
                h_i = th.squeeze(neighbour_h[:, i, :], 1)
                neighbour_r[:, i, :, :] = self.U_list[i](h_i).reshape(-1, self.rank, self.n_aggr)

            ris = None
            for i in range(self.n_aggr):
                if ris is None:
                    ris = self.U_output_list[i](self.G_list[i](neighbour_r[:, :, :, i], nodes))
                else:
                    aux = self.U_output_list[i](self.G_list[i](neighbour_r[:, :, :, i], nodes))
                    ris = th.cat((ris, aux), dim=1)

        return ris

