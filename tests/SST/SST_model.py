import torch.nn as nn
from treeLSTM import BinaryTreeLSTM


class SSTOutputModule(nn.module):

    def __init__(self,h_size, num_classes, dropout):
        super(SSTOutputModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)

    def forward(self, h):
        return self.linear(self.dropout(h))


def create_sst_model(num_vocabs,
                     x_size,
                     h_size,
                     num_classes,
                     dropout,
                     pretrained_emb=None,
                     cell_type='nary', **cell_args):

    input_module = nn.Embedding(num_vocabs, x_size)
    if pretrained_emb is not None:
        input_module.embedding.weight.data.copy_(pretrained_emb)
        input_module.embedding.weight.requires_grad = True

    output_module = SSTOutputModule(h_size, num_classes, dropout)

    m = BinaryTreeLSTM(x_size,h_size,input_module,output_module,cell_type,cell_args)

    return m
