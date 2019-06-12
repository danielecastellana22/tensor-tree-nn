import torch.nn as nn
import dgl


class TreeLSTM(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 input_module,
                 output_module,
                 cell):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.input_module = input_module
        self.output_module = output_module
        self.cell = cell

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
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # feed embedding
        # TODO: x * mask does not make sense. use x[mask]
        embeds = self.input_module(x * mask)
        g.ndata['iou_input'] = self.W_iou(embeds) * mask.float().unsqueeze(-1)
        g.ndata['f_input'] = self.W_f(embeds) * mask.float().unsqueeze(-1)
        # propagate
        dgl.prop_nodes_topo(g)
        # compute output
        #h = g.ndata.pop('h')
        h = g.ndata['h']
        out = self.output_module(h)
        return out
