import torch.nn as nn
import dgl
import dgl.init
from preprocessing.utils import ConstValues

class TreeModel(nn.Module):
    def __init__(self, x_size, h_size, input_module, output_module, cell_module, type_module=None):
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

        # apply input module and precompute its contribution
        x_mask = g.ndata['x'] != ConstValues.NO_ELEMENT
        x_embeds = self.input_module(g.ndata['x'] * x_mask)
        self.cell_module.precompute_input_values(g, x_embeds, x_mask)

        if self.type_module is not None:
            type_mask = g.ndata['type_id'] != ConstValues.NO_ELEMENT
            g.ndata['type_emb'] = self.type_module(g.ndata['type_id'] * type_mask)

        # propagate
        dgl.prop_nodes_topo(g)
        # compute output
        # h = g.ndata.pop('h')
        h = g.ndata['h']
        out = self.output_module(h)

        return out
