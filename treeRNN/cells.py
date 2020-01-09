import torch as th
import torch.nn as nn


class BaseCell(nn.Module):

    def __init__(self, h_size, max_output_degree, pos_stationarity):
        super(BaseCell, self).__init__()

        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity

    def forward(self, *input):
        pass

    def check_missing_children(self, neighbour_vals, bottom_val):
        n_missing = self.max_output_degree - neighbour_vals.size(1)
        if n_missing > 0:
            n_nodes = neighbour_vals.size(0)
            h_size = neighbour_vals.size(2)
            neighbour_vals = th.cat((neighbour_vals, bottom_val.reshape(1, 1, h_size).expand(n_nodes, n_missing, h_size)), dim=1)

        return neighbour_vals

    def message_func(self, edges):
        raise NotImplementedError("This function must be overrode!")

    def reduce_func(self, nodes):
        raise NotImplementedError("This function must be overrode!")

    def apply_node_func(self, nodes):
        raise NotImplementedError("This function must be overrode!")


class TreeLSTMCell(BaseCell):

    def __init__(self, h_size, max_output_degree, pos_stationarity, aggregator_class, **kwargs):
        super(TreeLSTMCell, self).__init__(h_size, max_output_degree, pos_stationarity)

        # TODO: a different bottom for each position
        # TODO: add parameter to choose to freeze or not the bottom values. Tensor or grad=False?
        self.bottom_h = nn.Parameter(th.zeros(h_size), requires_grad=True)
        self.bottom_c = nn.Parameter(th.zeros(h_size), requires_grad=True )

        # n_aggre = 3 because we would like to compute i,o,u gate
        self.aggregator_module = aggregator_class(h_size, max_output_degree, pos_stationarity, 3, **kwargs)

        if pos_stationarity:
            self.U_f = nn.Parameter(th.randn(h_size, h_size))
            self.b_f = nn.Parameter(th.randn(h_size))
        else:
            self.U_f_list = nn.ParameterList()
            self.b_f_list = nn.ParameterList()
            for i in range(max_output_degree):
                self.U_f_list.append(nn.Parameter(th.randn(h_size, h_size)))
                self.b_f_list.append(nn.Parameter(th.randn(h_size)))

    def compute_forget_gate(self, neighbour_h):
        if self.pos_stationarity:
            return th.addmm(self.b_f, neighbour_h.view(-1, self.h_size), self.U_f).view((-1, self.max_output_degree * self.h_size))
        else:
            ris = None
            for i in range(self.max_output_degree):
                U = self.U_f_list[i]
                b = self.b_f_list[i]
                h = neighbour_h[:, i, :].view(-1, self.h_size)
                if ris is not None:
                    ris = th.cat((ris, th.addmm(b, h, U)), dim=1)
                else:
                    ris = th.addmm(b, h, U)

            return ris

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        #check missin child
        neighbour_h = self.check_missing_children(nodes.mailbox['h'], self.bottom_h)
        neighbour_c = self.check_missing_children(nodes.mailbox['c'], self.bottom_c)

        # add the input contribution
        f_aggr = self.compute_forget_gate(neighbour_h) + (nodes.data['f_input']).repeat((1, self.max_output_degree))
        iou_aggr = self.aggregator_module(neighbour_h, nodes)

        f = th.sigmoid(f_aggr).view(*neighbour_c.size())
        c = th.sum(f * neighbour_c, 1)
        return {'iou_aggr': iou_aggr, 'c_aggr': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou_input']
        if 'iou_aggr' in nodes.data:
            # internal nodes
            iou += nodes.data['iou_aggr']

        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u

        if 'c_aggr' in nodes.data:
            # internal nodes
            c += nodes.data['c_aggr']

        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class TreeRNNCell(BaseCell):

    def __init__(self, h_size, max_output_degree, pos_stationarity, aggregator_class, **kwargs):
        super(TreeRNNCell, self).__init__(h_size, max_output_degree, pos_stationarity)

        # TODO: a different bottom for each position
        # TODO: add parameter to choose to freeze or not the bottom values. Tensor or grad=False?
        self.bottom_h = nn.Parameter(th.zeros(h_size), requires_grad=False)

        self.aggregator_module = aggregator_class(h_size, max_output_degree, pos_stationarity, 1, **kwargs)

    def message_func(self, edges):
        return {'h': edges.src['h']}

    def reduce_func(self, nodes):
        #check missing child
        neighbour_h = self.check_missing_children(nodes.mailbox['h'], self.bottom_h)

        h_aggr = self.aggregator_module(neighbour_h, nodes)

        return {'h_aggr': h_aggr}

    def apply_node_func(self, nodes):
        h = nodes.data['h_input']
        if 'h_aggr' in nodes.data:
            # internal nodes
            h += nodes.data['h_aggr']

        h = th.tanh(h)
        return {'h': h}