import torch as th
import torch.nn as nn
from .utils import LinearWithTypes


class BaseTreeCell(nn.Module):

    def __init__(self):
        super(BaseTreeCell, self).__init__()

    def forward(self, *input):
        pass

    def compute_child_aggregation(self, *args):
        pass
        #raise NotImplementedError("This function must be overridden!")

    def compute_node_states(self, *args):
        pass
        #raise NotImplementedError("This function must be overridden!")

    def message_func(self, edges):
        raise NotImplementedError("This function must be overridden!")

    def reduce_func(self, nodes):
        raise NotImplementedError("This function must be overridden!")

    def apply_node_func(self, nodes):
        raise NotImplementedError("This function must be overridden!")


class TreeLSTMCell(BaseTreeCell):

    def __init__(self, x_size, h_size, max_output_degree, pos_stationarity, aggregator_class, allow_input_labels=True, type_emb_size=None, **kwargs):
        super(TreeLSTMCell, self).__init__()

        self.x_size = x_size
        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity
        self.allow_input_labels = allow_input_labels

        if type_emb_size is not None:
            self.use_type_embs = True
        else:
            self.use_type_embs = False

        if self.max_output_degree > 0:
            # n_aggr = 3 because we would like to compute i,o,u gate
            self.aggregator_module = aggregator_class(h_size, self.max_output_degree, self.pos_stationarity,
                                                      type_emb_size=type_emb_size, n_aggr=3, **kwargs)
        else:
            self.aggregator_module = None

        # input matrices
        if self.allow_input_labels:
            # using a tensor to combine type embs and word embs is TOO SLOW
            # we ALWAYS ignoe type embs for the input
            self.iou_input_module = LinearWithTypes(x_size, 3 * h_size, None)
            self.forget_input_module = LinearWithTypes(x_size, h_size, None)

        # forget gate matrices
        if pos_stationarity:
            #self.forget_module = nn.Linear(h_size, h_size, bias=True)
            self.forget_module = LinearWithTypes(h_size, h_size, type_emb_size)
        else:
            self.forget_module = aggregator_class(h_size, max_output_degree, pos_stationarity,
                                                  type_emb_size=type_emb_size,
                                                  n_aggr=max_output_degree, **kwargs)

    def __compute_iou_input_values__(self, x, type_embs):
        # we ignore type embs
        type_embs = None
        if self.allow_input_labels:
            return self.iou_input_module(x, type_embs)
        else:
            raise ValueError('This cell cannot manage input labels!')

    def __compute_forget_input_values__(self, x, type_embs):
        # we ignore type embs
        type_embs = None
        if self.allow_input_labels:
            return self.forget_input_module(x, type_embs)
        else:
            raise ValueError('This cell cannot manage input labels!')

    def __compute_forget_gates__(self, x, neighbour_h, type_embs):
        n_batch = neighbour_h.size(0)
        n_ch = neighbour_h.size(1)
        # input computation does not depend on type embs
        f_input = self.__compute_forget_input_values__(x, type_embs).repeat(1, n_ch)
        if self.pos_stationarity:
            # TODO: this raise error because we have to expand also type_embs
            return self.forget_module(neighbour_h.view((-1, self.h_size)), type_embs).view(n_batch, n_ch * self.h_size) + f_input
        else:
            return self.forget_module(neighbour_h, type_embs) + f_input

    def compute_child_aggregation(self, x, n_h, n_c, type_embs):
        # add the input contribution
        f_aggr = self.__compute_forget_gates__(x, n_h, type_embs)
        iou_aggr = self.aggregator_module(n_h, type_embs)

        f = th.sigmoid(f_aggr).view(*n_c.size())
        c_aggr = th.sum(f * n_c, 1)
        return {'iou_aggr': iou_aggr, 'c_aggr': c_aggr}

    def compute_node_states(self, x, iou_aggr, c_aggr, type_embs):
        iou = self.__compute_iou_input_values__(x, type_embs)

        if iou_aggr is not None:
            # internal nodes
            iou += iou_aggr

        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u

        if c_aggr is not None:
            # internal nodes
            c += c_aggr

        h = o * th.tanh(c)
        return {'h': h, 'c': c}

    @classmethod
    def message_func(cls, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}  # , 'pos': edges.data['pos']}

    def reduce_func(self, nodes):
        x = nodes.data['x']
        neighbour_h = nodes.mailbox['h']
        neighbour_c = nodes.mailbox['c']
        type_embs = nodes.data['type_embs'] if self.use_type_embs else None

        return self.compute_child_aggregation(x, neighbour_h, neighbour_c, type_embs)

    def apply_node_func(self, nodes):
        x = nodes.data['x']
        iou_aggr = nodes.data['iou_aggr'] if 'iou_aggr' in nodes.data else None
        c_aggr = nodes.data['c_aggr'] if 'c_aggr' in nodes.data else None
        type_embs = nodes.data['type_embs'] if self.use_type_embs else None

        return self.compute_node_states(x, iou_aggr, c_aggr, type_embs)


# TODO: this class should work also with RNN cells!
class TypedTreeCell(BaseTreeCell):

    def __init__(self, x_size, h_size, cell_class, cells_params_list, share_input_matrices=False):
        super(TypedTreeCell, self).__init__()

        self.h_size = h_size
        self.cell_class = cell_class
        # number of different types
        self.n_types = len(cells_params_list)

        if share_input_matrices:
            raise NotImplementedError('Sharing input matrices is not implemented yet!')

        self.share_input_matrices = share_input_matrices

        self.n_aggr = 3
        self.cell_list = nn.ModuleList()

        for i in range(self.n_types):
            if self.share_input_matrices and i > 0:
                # if we are sharing input matrices, we use matrices of type = 0
                self.cell_list.append(cell_class(x_size, h_size, allow_input_labels=False, **cells_params_list[i]))
            else:
                self.cell_list.append(cell_class(x_size, h_size, **cells_params_list[i]))

    def message_func(self, edges):
        return self.cell_class.message_func(edges)

    def reduce_func(self, nodes):

        x = nodes.data['x']
        n_h = nodes.mailbox['h']
        n_c = nodes.mailbox['c']
        types = nodes.data['type_id']

        n_nodes = x.size(0)
        iou_aggr = th.zeros((n_nodes, self.n_aggr * self.h_size), device=x.device)
        c_aggr = th.zeros((n_nodes, self.h_size), device=x.device)

        for i in range(self.n_types):
            mask = (types == i)
            if th.sum(mask) > 0:
                ris = self.cell_list[i].compute_child_aggregation(x[mask, :], n_h[mask, :], n_c[mask, :], None)

                iou_aggr[mask, :] = ris['iou_aggr']
                c_aggr[mask, :] = ris['c_aggr']

        return {'iou_aggr': iou_aggr, 'c_aggr': c_aggr}

    def apply_node_func(self, nodes):
        x = nodes.data['x']
        iou_aggr = nodes.data['iou_aggr'] if 'iou_aggr' in nodes.data else None
        c_aggr = nodes.data['c_aggr'] if 'c_aggr' in nodes.data else None
        types = nodes.data['type_id']

        n_nodes = x.size(0)
        h = th.zeros((n_nodes, self.h_size), device=x.device)
        c = th.zeros((n_nodes, self.h_size), device=x.device)

        for i in range(self.n_types):
            mask = (types == i)
            if th.sum(mask) > 0:
                if iou_aggr is not None and c_aggr is not None:
                    ris = self.cell_list[i].compute_node_states(x[mask, :], iou_aggr[mask, :], c_aggr[mask, :], None)
                else:
                    ris = self.cell_list[i].compute_node_states(x[mask, :], None, None, None)

                h[mask, :] = ris['h']
                c[mask, :] = ris['c']

        return {'h': h, 'c': c}


# TODO: update the code of treeRNNCell according to the new refactor
class TreeRNNCell(BaseTreeCell):

    def __init__(self, x_size, h_size, **kwargs):
        super(TreeRNNCell, self).__init__(x_size, h_size)

        # input matrices
        self.W_in = nn.Linear(x_size, h_size, bias=True)

        self.aggregator_module = aggregator_class(h_size, max_output_degree, pos_stationarity, 1, **kwargs)

    def compute_input_values(self, g, x, mask):
        g.ndata['h_input'] = self.W_in(x) * mask.float().unsqueeze(-1)

    def message_func(self, edges):
        return {'h': edges.src['h']}

    def reduce_func(self, nodes):
        neighbour_h = nodes.mailbox['h']

        h_aggr = self.aggregator_module(neighbour_h, nodes)

        return {'h_aggr': h_aggr}

    def apply_node_func(self, nodes):
        h = nodes.data['h_input']
        if 'h_aggr' in nodes.data:
            # internal nodes
            h += nodes.data['h_aggr']

        h = th.tanh(h)
        return {'h': h}

