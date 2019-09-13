import torch.nn as nn
import dgl
from .cells import TreeLSTMCell, TreeRNNCell


class BaseTreeModel(nn.Module):

    def __init__(self, x_size, h_size, max_output_degree, pos_stationarity, input_module, output_module):
        super(BaseTreeModel, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.max_output_degree = max_output_degree
        self.pos_stationarity = pos_stationarity
        self.input_module = input_module
        self.output_module = output_module

    def forward(self, g, x, mask):
        pass


class TreeLSTM(BaseTreeModel):
    def __init__(self,  x_size, h_size, max_output_degree, pos_stationarity, input_module, output_module,
                 aggregator_class, **kwargs):
        super(TreeLSTM, self).__init__(x_size, h_size, max_output_degree, pos_stationarity, input_module, output_module)

        self.cell_module = TreeLSTMCell(h_size, max_output_degree, pos_stationarity, aggregator_class, **kwargs)
        # input matrices
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=True)
        self.W_f = nn.Linear(x_size, h_size, bias=True)

    def forward(self, g, x, mask):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : TreeDataset.TreeBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        g.register_message_func(self.cell_module.message_func)
        g.register_reduce_func(self.cell_module.reduce_func)
        g.register_apply_node_func(self.cell_module.apply_node_func)
        # feed embedding
        # TODO: x * mask assume zero neutral element. THIS NO TRUE IF THE INPUT CONTRIBUTION IS NOT SUMMED
        embeds = self.input_module(x * mask)
        g.ndata['x'] = embeds
        g.ndata['iou_input'] = self.W_iou(embeds) * mask.float().unsqueeze(-1)
        g.ndata['f_input'] = self.W_f(embeds) * mask.float().unsqueeze(-1)
        # propagate
        dgl.prop_nodes_topo(g)
        # compute output
        # h = g.ndata.pop('h')
        h = g.ndata['h']
        out = self.output_module(h)
        return out


class TreeRNN(BaseTreeModel):

    def __init__(self,  x_size, h_size, max_output_degree, pos_stationarity, input_module, output_module,
                 aggregator_class, **kwargs):
        super(TreeRNN, self).__init__(x_size, h_size, max_output_degree, pos_stationarity, input_module, output_module)

        self.cell_module = TreeRNNCell(h_size, max_output_degree, pos_stationarity, aggregator_class, **kwargs)
        # input matrices
        self.W_in = nn.Linear(x_size, h_size, bias=True)

    def forward(self, g, x, mask):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : TreeDataset.TreeBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        g.register_message_func(self.cell_module.message_func)
        g.register_reduce_func(self.cell_module.reduce_func)
        g.register_apply_node_func(self.cell_module.apply_node_func)
        # feed embedding
        # TODO: x * mask assume zero neutral element. THIS NO TRUE IF THE INPUT CONTRIBUTION IS NOT SUMMED
        embeds = self.input_module(x * mask)
        g.ndata['x'] = embeds
        g.ndata['h_input'] = self.W_in(embeds) * mask.float().unsqueeze(-1)
        # propagate
        dgl.prop_nodes_topo(g)
        # compute output
        # h = g.ndata.pop('h')
        h = g.ndata['h']
        out = self.output_module(h)
        return out
