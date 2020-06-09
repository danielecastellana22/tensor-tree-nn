import torch.nn as nn
import dgl
import dgl.init


class TreeModel(nn.Module):
    def __init__(self, input_module, output_module, cell_module, type_module, only_root_state=False):
        super(TreeModel, self).__init__()
        self.input_module = input_module
        self.output_module = output_module
        self.type_module = type_module
        self.cell_module = cell_module

        self.only_root_state = only_root_state

    def forward(self, *t_list):
        out_list = []
        for t in t_list:
            t.set_n_initializer(dgl.init.zero_initializer)
            t.register_message_func(self.cell_module.message_func)
            t.register_reduce_func(self.cell_module.reduce_func)
            t.register_apply_node_func(self.cell_module.apply_node_func)

            # apply type module
            if self.type_module is not None:
                type_mask = t.ndata['t_mask']
                t.ndata['t_embs'] = self.type_module(t.ndata['t'] * type_mask) * type_mask.view(-1, 1)

            # apply input module
            if self.input_module is not None:
                x_mask = t.ndata['x_mask']
                t.ndata['x_embs'] = self.input_module(t.ndata['x'] * x_mask) * x_mask.view(-1, 1)

            # propagate
            dgl.prop_nodes_topo(t)

            # return the hidden
            h = t.ndata['h']

            if self.only_root_state:
                root_ids = [i for i in range(t.number_of_nodes()) if t.out_degree(i) == 0]
                out_list.append(h[root_ids])
            else:
                out_list.append(h)

        if self.output_module is not None:
            return self.output_module(*out_list)
        else:
            return out_list