import torch.nn as nn
import dgl


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

        g.register_message_func(self.cell_module.message_func)
        g.register_reduce_func(self.cell_module.reduce_func)
        g.register_apply_node_func(self.cell_module.apply_node_func)

        # apply input module and precompute its contribution
        embeds = self.input_module(g.ndata['x'] * g.ndata['mask']) #* mask.unsqueeze(-1).float()
        self.cell_module.precompute_input_values(g, embeds)

        if self.type_module:
            g.ndata['type_emb'] = self.type_module(g.ndata['type_id'])

        # propagate
        dgl.prop_nodes_topo(g)
        # compute output
        # h = g.ndata.pop('h')
        h = g.ndata['h']
        out = self.output_module(h)

        return out
