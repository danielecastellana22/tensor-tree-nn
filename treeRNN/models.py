import torch.nn as nn
import dgl
import dgl.init
from preprocessing.utils import ConstValues


class TreeModel(nn.Module):
    def __init__(self, x_size, h_size, input_module, output_module, cell_module, type_module):
        super(TreeModel, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.input_module = input_module
        self.output_module = output_module
        self.type_module = type_module

        self.cell_module = cell_module

    def forward(self, g):

        g.set_n_initializer(dgl.init.zero_initializer)
        g.register_message_func(self.cell_module.message_func)
        g.register_reduce_func(self.cell_module.reduce_func)
        g.register_apply_node_func(self.cell_module.apply_node_func)

        # apply type module
        if self.type_module is not None:
            type_mask = g.ndata['t_mask']
            g.ndata['t'] = self.type_module(g.ndata['t'] * type_mask) * type_mask.view(-1, 1)

        # apply input module
        if self.input_module is not None:
            x_mask = g.ndata['x_mask']
            g.ndata['x'] = self.input_module(g.ndata['x'] * x_mask) * x_mask.view(-1, 1)

        # propagate
        dgl.prop_nodes_topo(g)

        # compute output
        h = g.ndata['h']
        if self.output_module is not None:
            return self.output_module(h)
        else:
            return h
