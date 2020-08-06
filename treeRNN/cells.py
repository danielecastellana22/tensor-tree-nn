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


class TreeLSTMCell(BaseTreeCell):

    def __init__(self, x_size, h_size, aggregator_class, pos_stationarity=False, max_output_degree=0, type_emb_size=None, **kwargs):
        super(TreeLSTMCell, self).__init__()

        self.x_size = x_size
        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity
        self.type_emb_size = type_emb_size

        if type_emb_size is not None:
            self.use_type_embs = True
        else:
            self.use_type_embs = False

        #h_size, pos_stationarity, max_output_degree, type_emb_size, n_aggr):
        # n_aggr = 3 because we would like to compute i,o,u gate
        self.aggregator_module = aggregator_class(h_size, pos_stationarity, max_output_degree, type_emb_size,
                                                  n_aggr=3, **kwargs)


        # we ALWAYS ignoe type embs for the input
        self.iou_input_module = nn.Linear(x_size, 3 * h_size, bias=True)
        self.forget_input_module = nn.Linear(x_size, h_size, bias=True)

        # forget gate matrices
        if pos_stationarity:
            if self.use_type_embs:
                self.forget_module = aggregator_class(h_size, pos_stationarity=False, max_output_degree=1,
                                                      type_emb_size=type_emb_size,
                                                      n_aggr=1, **kwargs)
            else:
                self.forget_module = nn.Linear(h_size, h_size, bias= True)
        else:
            self.forget_module = aggregator_class(h_size, pos_stationarity, max_output_degree, type_emb_size,
                                                  n_aggr=max_output_degree, **kwargs)

    def __compute_forget_gates__(self, x, x_mask, neighbour_h, type_embs):
        n_batch = neighbour_h.size(0)
        n_ch = neighbour_h.size(1)
        # input computation does not depend on type embs
        f_input = self.forget_input_module(x) * x_mask.view(-1, 1)
        f_input = f_input.repeat(1, n_ch)
        if self.pos_stationarity:
            if self.use_type_embs:
                f_gate = self.forget_module(neighbour_h.view((-1, 1, self.h_size)), type_embs.repeat(1, n_ch).reshape(-1, self.type_emb_size)).view(n_batch, n_ch * self.h_size)
            else:
                f_gate = self.forget_module(neighbour_h.view((-1, self.h_size))).view(n_batch, n_ch * self.h_size)
        else:
            f_gate = self.forget_module(neighbour_h, type_embs) + f_input

        return f_gate + f_input

    @classmethod
    def message_func(cls, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}  # , 'pos': edges.data['pos']}

    def reduce_func(self, nodes):
        x = nodes.data['x_embs']
        x_mask = nodes.data['x_mask']
        neighbour_h = nodes.mailbox['h']
        neighbour_c = nodes.mailbox['c']
        type_embs = nodes.data['t_embs'] if self.use_type_embs else None

        # add the input contribution
        f_aggr = self.__compute_forget_gates__(x, x_mask, neighbour_h, type_embs)
        iou_aggr = self.aggregator_module(neighbour_h, type_embs)

        f = th.sigmoid(f_aggr).view(*neighbour_c.size())
        c_aggr = th.sum(f * neighbour_c, 1)
        return {'iou_aggr': iou_aggr, 'c_aggr': c_aggr}

    def apply_node_func(self, nodes):
        x = nodes.data['x_embs']
        x_mask = nodes.data['x_mask']
        iou_aggr = nodes.data['iou_aggr'] if 'iou_aggr' in nodes.data else None
        c_aggr = nodes.data['c_aggr'] if 'c_aggr' in nodes.data else None

        iou = self.iou_input_module(x) * x_mask.view(-1, 1)

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


class TreeNetCell(BaseTreeCell):

    def __init__(self, x_size, h_size, aggregator_class, pos_stationarity=False, max_output_degree=0, type_emb_size=None, **kwargs):
        super(TreeNetCell, self).__init__()

        self.x_size = x_size
        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity
        self.type_emb_size = type_emb_size

        if type_emb_size is not None:
            self.use_type_embs = True
        else:
            self.use_type_embs = False

        # h_size, pos_stationarity, max_output_degree, type_emb_size, n_aggr):
        # n_aggr = 3 because we would like to compute i,o,u gate
        self.aggregator_module = aggregator_class(h_size, pos_stationarity, max_output_degree, type_emb_size,
                                                  n_aggr=1, **kwargs)


        # we ALWAYS ignoe type embs for the input
        self.iou_input_module = nn.Linear(x_size, 3 * h_size, bias=True)
        self.forget_input_module = nn.Linear(x_size, h_size, bias=True)

        # forget gate matrices
        if pos_stationarity:
            if self.use_type_embs:
                self.forget_module = aggregator_class(h_size, pos_stationarity=False, max_output_degree=1,
                                                      type_emb_size=type_emb_size,
                                                      n_aggr=1, **kwargs)
            else:
                self.forget_module = nn.Linear(h_size, h_size, bias= True)
        else:
            self.forget_module = aggregator_class(h_size, pos_stationarity, max_output_degree, type_emb_size,
                                                  n_aggr=max_output_degree, **kwargs)

    def __compute_forget_gates__(self, x, x_mask, neighbour_h, type_embs):
        n_batch = neighbour_h.size(0)
        n_ch = neighbour_h.size(1)
        # input computation does not depend on type embs
        f_input = self.forget_input_module(x) * x_mask.view(-1, 1)
        #assert th.sum(x_mask.view(-1)).item() == 0
        f_input = f_input.repeat(1, n_ch)
        if self.pos_stationarity:
            if self.use_type_embs:
                f_gate = self.forget_module(neighbour_h.view((-1, 1, self.h_size)), type_embs.repeat(1, n_ch).reshape(-1, self.type_emb_size)).view(n_batch, n_ch * self.h_size)
            else:
                f_gate = self.forget_module(neighbour_h.view((-1, self.h_size))).view(n_batch, n_ch * self.h_size)
        else:
            f_gate = self.forget_module(neighbour_h, type_embs) + f_input

        return f_gate + f_input

    @classmethod
    def message_func(cls, edges):
        return {'h': edges.src['h'], 'c': edges.src['c'], 'pos': edges.data['pos']}

    def reduce_func(self, nodes):
        x = nodes.data['x_embs']
        x_mask = nodes.data['x_mask']

        neighbour_h = nodes.mailbox['h']
        neighbour_c = nodes.mailbox['c']
        pos = nodes.mailbox['pos'].unsqueeze(2).expand_as(neighbour_h)

        neighbour_h = th.gather(neighbour_h, 1, pos)
        neighbour_c = th.gather(neighbour_c, 1, pos)

        type_embs = nodes.data['t_embs'] if self.use_type_embs else None

        # add the input contribution
        f_aggr = self.__compute_forget_gates__(x, x_mask, neighbour_h, type_embs)
        iou_aggr = self.aggregator_module(neighbour_h, type_embs)

        f = th.sigmoid(f_aggr).view(*neighbour_c.size())
        c_aggr = th.sum(f * neighbour_c, 1)
        return {'iou_aggr': iou_aggr, 'c_aggr': c_aggr}

    def apply_node_func(self, nodes):
        x = nodes.data['x_embs']
        x_mask = nodes.data['x_mask']
        iou_aggr = nodes.data['iou_aggr'] if 'iou_aggr' in nodes.data else None
        c_aggr = nodes.data['c_aggr'] if 'c_aggr' in nodes.data else None

        if c_aggr is None:
            # leaf
            iou = self.iou_input_module(x) * x_mask.view(-1, 1)

            i, o, u = th.chunk(iou, 3, 1)
            i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
            c = i * u

        else:
            #internal
            # internal nodes
            c = c_aggr
            o = iou_aggr

        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class TreeRNNCell(BaseTreeCell):

    def __init__(self, x_size, h_size, aggregator_class, pos_stationarity=False, max_output_degree=0, type_emb_size=None, **kwargs):
        super(TreeRNNCell, self).__init__()

        self.x_size = x_size
        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity
        self.type_emb_size = type_emb_size

        if type_emb_size is not None:
            self.use_type_embs = True
        else:
            self.use_type_embs = False

        self.aggregator_module = aggregator_class(h_size, self.pos_stationarity, self.max_output_degree,
                                                  type_emb_size, n_aggr=1, **kwargs)

        # we ALWAYS ignoe type embs for the input
        self.input_module = nn.Linear(x_size, h_size, bias=True)

    @classmethod
    def message_func(cls, edges):
        return {'h': edges.src['h']}  # , 'pos': edges.data['pos']}

    def reduce_func(self, nodes):
        neighbour_h = nodes.mailbox['h']
        type_embs = nodes.data['t_embs'] if self.use_type_embs else None

        h_aggr = self.aggregator_module(neighbour_h, type_embs)
        return {'h_aggr': h_aggr}

    def apply_node_func(self, nodes):
        x = nodes.data['x_embs']
        x_mask = nodes.data['x_mask']
        h_aggr = nodes.data['h_aggr'] if 'h_aggr'  in nodes.data else None

        h = self.input_module(x) * x_mask.view(-1, 1)
        if h_aggr is not None:
            # internal nodes
            h += h_aggr
        h = th.tanh(h)

        return {'h': h}


# TODO: this class must ber rewritten! should work also with RNN cells!
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

        x = nodes.data['x_embs']
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
        x = nodes.data['x_embs']
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

