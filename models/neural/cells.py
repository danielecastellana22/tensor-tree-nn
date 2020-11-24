import torch as th
import torch.nn as nn
from utils.misc import string2class
from experiments.config import create_object_from_config


class BaseCell(nn.Module):

    def __init__(self):
        super(BaseCell, self).__init__()

    def forward(self, *input):
        pass

    def message_func(self, edges):
        raise NotImplementedError("This function must be overridden!")

    def reduce_func(self, nodes, type_mask=None):
        raise NotImplementedError("This function must be overridden!")

    def apply_node_func(self, nodes, type_mask=None):
        raise NotImplementedError("This function must be overridden!")


class LSTM(BaseCell):

    def __init__(self, x_size, h_size, aggregator_class, pos_stationarity=False, max_output_degree=0, t_size=None, **kwargs):
        super(LSTM, self).__init__()

        self.x_size = x_size
        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity
        self.t_size = t_size

        if t_size is not None:
            self.use_types = True
        else:
            self.use_types = False

        # n_aggr = 3 because we would like to compute i,o,u gate
        aggregator_class = string2class(aggregator_class)
        self.aggregator_module = aggregator_class(h_size, pos_stationarity, max_output_degree, t_size,
                                                  n_aggr=3, **kwargs)

        # we ALWAYS ignoe type embs for the input
        self.iou_input_module = nn.Linear(x_size, 3 * h_size, bias=True)
        self.forget_input_module = nn.Linear(x_size, h_size, bias=True)

        # forget gate matrices
        if pos_stationarity:
            if self.use_types:
                self.forget_module = aggregator_class(h_size, pos_stationarity=False, max_output_degree=1,
                                                      type_emb_size=t_size,
                                                      n_aggr=1, **kwargs)
            else:
                self.forget_module = nn.Linear(h_size, h_size, bias= True)
        else:
            self.forget_module = aggregator_class(h_size, pos_stationarity, max_output_degree, t_size,
                                                  n_aggr=max_output_degree, **kwargs)

    def __compute_forget_gates__(self, x, x_mask, neighbour_h, type_embs):
        n_batch = neighbour_h.size(0)
        n_ch = neighbour_h.size(1)
        # input computation does not depend on type embs
        f_input = self.forget_input_module(x) * x_mask.view(-1, 1)
        f_input = f_input.repeat(1, n_ch)
        if self.pos_stationarity:
            if self.use_types:
                f_gate = self.forget_module(neighbour_h.view((-1, 1, self.h_size)), type_embs.repeat(1, n_ch).reshape(-1, self.t_size)).view(n_batch, n_ch * self.h_size)
            else:
                f_gate = self.forget_module(neighbour_h.view((-1, self.h_size))).view(n_batch, n_ch * self.h_size)
        else:
            f_gate = self.forget_module(neighbour_h, type_embs)

        return f_gate[:, :n_ch*self.h_size] + f_input

    @classmethod
    def message_func(cls, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes, type_mask = None):
        if type_mask is None:
            x = nodes.data['x_embs']
            x_mask = nodes.data['x_mask']
            neighbour_h = nodes.mailbox['h']
            neighbour_c = nodes.mailbox['c']
            type_embs = nodes.data['t_embs'] if self.use_types else None
        else:
            x = nodes.data['x_embs'][type_mask]
            x_mask = nodes.data['x_mask'][type_mask]
            neighbour_h = nodes.mailbox['h'][type_mask]
            neighbour_c = nodes.mailbox['c'][type_mask]
            type_embs = None

        # add the input contribution
        f_aggr = self.__compute_forget_gates__(x, x_mask, neighbour_h, type_embs)
        iou_aggr = self.aggregator_module(neighbour_h, type_embs)

        f = th.sigmoid(f_aggr).view(*neighbour_c.size())
        c_aggr = th.sum(f * neighbour_c, 1)
        return {'iou_aggr': iou_aggr, 'c_aggr': c_aggr}

    def apply_node_func(self, nodes, type_mask=None):
        if type_mask is None:
            x = nodes.data['x_embs']
            x_mask = nodes.data['x_mask']
            iou_aggr = nodes.data['iou_aggr'] if 'iou_aggr' in nodes.data else None
            c_aggr = nodes.data['c_aggr'] if 'c_aggr' in nodes.data else None
        else:
            x = nodes.data['x_embs'][type_mask]
            x_mask = nodes.data['x_mask'][type_mask]
            iou_aggr = nodes.data['iou_aggr'][type_mask] if 'iou_aggr' in nodes.data else None
            c_aggr = nodes.data['c_aggr'][type_mask] if 'c_aggr' in nodes.data else None

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


class TreeNet(BaseCell):

    def __init__(self, x_size, h_size, aggregator_class, pos_stationarity=False, max_output_degree=0, t_size=None, **kwargs):
        super(TreeNet, self).__init__()

        self.x_size = x_size
        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity
        self.t_size = t_size

        if t_size is not None:
            self.use_types = True
        else:
            self.use_types = False

        aggregator_class = string2class(aggregator_class)
        self.aggregator_module = aggregator_class(h_size, pos_stationarity, max_output_degree, t_size,
                                                  n_aggr=1, **kwargs)

        # we ALWAYS ignoe type embs for the input
        self.iou_input_module = nn.Linear(x_size, 3 * h_size, bias=True)
        self.forget_input_module = nn.Linear(x_size, h_size, bias=True)

        # forget gate matrices
        if pos_stationarity:
            if self.use_types:
                self.forget_module = aggregator_class(h_size, pos_stationarity=False, max_output_degree=1,
                                                      type_emb_size=t_size,
                                                      n_aggr=1, **kwargs)
            else:
                self.forget_module = nn.Linear(h_size, h_size, bias= True)
        else:
            self.forget_module = aggregator_class(h_size, pos_stationarity, max_output_degree, t_size,
                                                  n_aggr=max_output_degree, **kwargs)

    def __compute_forget_gates__(self, x, x_mask, neighbour_h, type_embs):
        n_batch = neighbour_h.size(0)
        n_ch = neighbour_h.size(1)
        # input computation does not depend on type embs
        f_input = self.forget_input_module(x) * x_mask.view(-1, 1)
        #assert th.sum(x_mask.view(-1)).item() == 0
        f_input = f_input.repeat(1, n_ch)
        if self.pos_stationarity:
            if self.use_types:
                f_gate = self.forget_module(neighbour_h.view((-1, 1, self.h_size)), type_embs.repeat(1, n_ch).reshape(-1, self.t_size)).view(n_batch, n_ch * self.h_size)
            else:
                f_gate = self.forget_module(neighbour_h.view((-1, self.h_size))).view(n_batch, n_ch * self.h_size)
        else:
            f_gate = self.forget_module(neighbour_h, type_embs) + f_input

        return f_gate + f_input

    @classmethod
    def message_func(cls, edges):
        return {'h': edges.src['h'], 'c': edges.src['c'], 'pos': edges.data['pos']}

    def reduce_func(self, nodes, type_mask=None):
        if type_mask is None:
            x = nodes.data['x_embs']
            x_mask = nodes.data['x_mask']
            neighbour_h = nodes.mailbox['h']
            neighbour_c = nodes.mailbox['c']
            pos = nodes.mailbox['pos'].unsqueeze(2).expand_as(neighbour_h)
            type_embs = nodes.data['t_embs'] if self.use_types else None
        else:
            x = nodes.data['x_embs'][type_mask]
            x_mask = nodes.data['x_mask'][type_mask]
            neighbour_h = nodes.mailbox['h'][type_mask]
            neighbour_c = nodes.mailbox['c'][type_mask]
            pos = nodes.mailbox['pos'][type_mask].unsqueeze(2).expand_as(neighbour_h)
            type_embs = nodes.data['t_embs'][type_mask] if self.use_types else None

        neighbour_h = th.gather(neighbour_h, 1, pos)
        neighbour_c = th.gather(neighbour_c, 1, pos)

        # add the input contribution
        f_aggr = self.__compute_forget_gates__(x, x_mask, neighbour_h, type_embs)
        iou_aggr = self.aggregator_module(neighbour_h, type_embs)

        f = th.sigmoid(f_aggr).view(*neighbour_c.size())
        c_aggr = th.sum(f * neighbour_c, 1)
        return {'iou_aggr': iou_aggr, 'c_aggr': c_aggr}

    def apply_node_func(self, nodes, type_mask=None):
        if type_mask is None:
            x = nodes.data['x_embs']
            x_mask = nodes.data['x_mask']
            iou_aggr = nodes.data['iou_aggr'] if 'iou_aggr' in nodes.data else None
            c_aggr = nodes.data['c_aggr'] if 'c_aggr' in nodes.data else None
        else:
            x = nodes.data['x_embs'][type_mask]
            x_mask = nodes.data['x_mask'][type_mask]
            iou_aggr = nodes.data['iou_aggr'][type_mask] if 'iou_aggr' in nodes.data else None
            c_aggr = nodes.data['c_aggr'][type_mask] if 'c_aggr' in nodes.data else None

        if c_aggr is None:
            # leaf
            iou = self.iou_input_module(x) * x_mask.view(-1, 1)

            i, o, u = th.chunk(iou, 3, 1)
            i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
            c = i * u

        else:
            # internal nodes
            c = c_aggr
            o = iou_aggr

        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class RNN(BaseCell):

    def __init__(self, x_size, h_size, aggregator_class, pos_stationarity=False, max_output_degree=0, t_size=None, **kwargs):
        super(RNN, self).__init__()

        self.x_size = x_size
        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity
        self.t_size = t_size

        if t_size is not None:
            self.use_types = True
        else:
            self.use_types = False

        aggregator_class = string2class(aggregator_class)
        self.aggregator_module = aggregator_class(h_size, self.pos_stationarity, self.max_output_degree,
                                                  t_size, n_aggr=1, **kwargs)

        # we ALWAYS ignoe type embs for the input
        self.input_module = nn.Linear(x_size, h_size, bias=True)

    @classmethod
    def message_func(cls, edges):
        return {'h': edges.src['h']}  # , 'pos': edges.data['pos']}

    def reduce_func(self, nodes, type_mask=None):
        if type_mask is None:
            neighbour_h = nodes.mailbox['h']
            type_embs = nodes.data['t_embs'] if self.use_types else None
        else:
            neighbour_h = nodes.mailbox['h'][type_mask]
            type_embs = nodes.data['t_embs'][type_mask] if self.use_types else None

        h_aggr = self.aggregator_module(neighbour_h, type_embs)
        return {'h_aggr': h_aggr}

    def apply_node_func(self, nodes, type_mask=None):
        if type_mask is None:
            x = nodes.data['x_embs']
            x_mask = nodes.data['x_mask']
            h_aggr = nodes.data['h_aggr'] if 'h_aggr' in nodes.data else None
        else:
            x = nodes.data['x_embs'][type_mask]
            x_mask = nodes.data['x_mask'][type_mask]
            h_aggr = nodes.data['h_aggr'][type_mask] if 'h_aggr' in nodes.data else None

        h = self.input_module(x) * x_mask.view(-1, 1)
        if h_aggr is not None:
            # internal nodes
            h += h_aggr
        h = th.tanh(h)

        return {'h': h}


class Typed(BaseCell):

    # TODO: we can use a list of params to handle different aggregators for each type
    # TODO: how can we share input matrices among types?
    def __init__(self, x_size, h_size, num_types, cell_config):
        super(Typed, self).__init__()

        self.h_size = h_size
        # number of different types
        self.num_types = num_types

        self.cell_list = nn.ModuleList()

        for i in range(self.num_types):
            self.cell_list.append(create_object_from_config(cell_config, h_size=h_size, x_size=x_size))
        # append another cell for nodes with no type
        self.cell_list.append(create_object_from_config(cell_config, h_size=h_size, x_size=x_size))

    def message_func(self, edges):
        return self.cell_list[0].message_func(edges)

    def reduce_func(self, nodes, type_mask=None):
        types = nodes.data['t']
        n_nodes = types.size(0)
        out_ris = {}

        for i in range(-1, self.num_types):
            mask = (types == i)
            if th.any(mask):
                ris = self.cell_list[i].reduce_func(nodes, type_mask=mask)
                for k, v in ris.items():
                    if k not in out_ris:
                        out_ris[k] = th.zeros((n_nodes, v.shape[1]), device=v.device)
                    out_ris[k][mask] = v

        return out_ris

    def apply_node_func(self, nodes, type_mask=None):
        types = nodes.data['t']
        n_nodes = types.size(0)
        out_ris = {}

        for i in range(-1, self.num_types):
            mask = (types == i)
            if th.any(mask):
                ris = self.cell_list[i].apply_node_func(nodes, type_mask=mask)
                for k, v in ris.items():
                    if k not in out_ris:
                        out_ris[k] = th.zeros((n_nodes, v.shape[1]), device=v.device)
                    out_ris[k][mask] = v

        return out_ris

