import torch as th
import torch.nn as nn


class BaseTreeCell(nn.Module):

    def __init__(self):
        super(BaseTreeCell, self).__init__()

    def forward(self, *input):
        pass

    def message_func(self, edges):
        raise NotImplementedError("This function must be overridden!")

    def reduce_func(self, nodes):
        raise NotImplementedError("This function must be overridden!")

    def apply_node_func(self, nodes):
        raise NotImplementedError("This function must be overridden!")

    def precompute_input_values(self, g, x):
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

        # TODO: input matrices  must depends on type_emb_size
        # input matrices
        if self.allow_input_labels:
            self.iou_input_module = nn.Linear(x_size, 3 * h_size, bias=True)
            self.forget_input_module = nn.Linear(x_size, h_size, bias=True)

        # forget gate matrices
        if pos_stationarity:
            # TODO: can we use aggregator in order to make advantage of type embs?
            self.forget_module = nn.Linear(h_size, h_size, bias=True)
        else:
            self.forget_module = aggregator_class(h_size, max_output_degree, pos_stationarity,
                                                  type_emb_size=type_emb_size, n_aggr=max_output_degree, **kwargs)

    def apply_input_matrices(self, x):
        if self.allow_input_labels:
            return {'iou_input': self.iou_input_module(x), 'f_input': self.forget_input_module(x)}
        else:
            raise ValueError('This cell cannot manage input labels!')

    def compute_forget_gate(self, neighbour_h, type_embs):
        n_batch = neighbour_h.size(0)
        n_ch = neighbour_h.size(1)

        if self.pos_stationarity:
            # TODO: what about type embs?
            return self.forget_module(neighbour_h.view((-1, self.h_size))).view(n_batch, n_ch * self.h_size)
        else:
            return self.forget_module(neighbour_h, type_embs)

    def aggregate_child_messages(self, neighbour_h, neighbour_c, f_input, type_embs):
        n_ch = neighbour_h.size(1)

        # add the input contribution
        f_aggr = self.compute_forget_gate(neighbour_h, type_embs) + f_input.repeat((1, n_ch))
        iou_aggr = self.aggregator_module(neighbour_h, type_embs)

        f = th.sigmoid(f_aggr).view(*neighbour_c.size())
        c_aggr = th.sum(f * neighbour_c, 1)
        return {'iou_aggr': iou_aggr, 'c_aggr': c_aggr}

    def precompute_input_values(self, g, x):
        ris = self.apply_input_matrices(x)
        mask = g.ndata['mask'].unsqueeze(-1).float()
        for (k, v) in ris.items():
            g.ndata[k] = v * mask

    @classmethod
    def message_func(cls, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}  # , 'pos': edges.data['pos']}

    def reduce_func(self, nodes):
        # pos = nodes.mailbox['pos']
        # aux = th.sum(pos, dim=0).numpy()
        # assert aux[0] == 0 and aux[1] == pos.size(0)
        if self.use_type_embs:
            return self.aggregate_child_messages(nodes.mailbox['h'], nodes.mailbox['c'], nodes.data['f_input'], nodes.data['type_emb'])
        else:
            return self.aggregate_child_messages(nodes.mailbox['h'], nodes.mailbox['c'], nodes.data['f_input'], None)

    @classmethod
    def apply_node_func(cls, nodes):
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


class TypedTreeCell(BaseTreeCell):

    def __init__(self, x_size, h_size, cell_class, cells_params_list, share_input_matrices=False):
        super(TypedTreeCell, self).__init__()

        self.h_size = h_size
        self.cell_class = cell_class
        # number of different types
        self.n_types = len(cells_params_list)

        self.share_input_matrices = share_input_matrices
        self.n_aggr = 3
        self.cell_list = nn.ModuleList()

        for i in range(self.n_types):
            if self.share_input_matrices and i > 0:
                # if we are sharing input matrices, we use matrices of type = 0
                self.cell_list.append(cell_class(x_size, h_size, allow_input_labels=False, **cells_params_list[i]))
            else:
                self.cell_list.append(cell_class(x_size, h_size, **cells_params_list[i]))

    def precompute_input_values(self, g, x):
        if self.share_input_matrices:
            self.cell_list[0].precompute_input_values(g, x)
        else:
            mask = g.ndata['mask'].bool()
            for i in range(self.n_types):
                type_mask = (g.ndata['type_id'] == i) * mask
                if th.sum(type_mask) > 0:
                    ris = self.cell_list[i].apply_input_matrices(x[type_mask])
                    for (k, v) in ris.items():
                        if k not in g.ndata:
                            g.ndata[k] = th.zeros((x.size(0), v.size(1)),  device=x.device)
                        g.ndata[k][type_mask] = v

    def message_func(self, edges):
        return self.cell_class.message_functions(edges)

    def reduce_func(self, nodes):
        n_h = nodes.mailbox['h']
        n_c = nodes.mailbox['c']
        f_in = nodes.data['f_input']
        types = nodes.data['type_id']

        n_nodes = n_h.size(0)
        iou_aggr = th.zeros((n_nodes, self.n_aggr * self.h_size), device=n_h.device)
        c_aggr = th.zeros((n_nodes, self.h_size), device=n_h.device)

        for i in range(self.n_types):
            mask = (types == i)
            if th.sum(mask) > 0:
                ris = self.cell_list[i].aggregate_child_messages(n_h[mask, :, :],
                                                                 n_c[mask, :, :],
                                                                 f_in[mask],
                                                                 None)
                iou_aggr[mask, :] = ris['iou_aggr']
                c_aggr[mask, :] = ris['c_aggr']

        return {'iou_aggr': iou_aggr, 'c_aggr': c_aggr}

    def apply_node_func(self, nodes):
        return self.cell_class.node_computation(nodes)


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

