from utils.utils import string2class
import torch.nn as nn


class OneLayerNN(nn.Module):

    def __init__(self, in_size, num_classes, dropout, h_size=0, non_linearity='torch.nn.ReLU'):
        super(OneLayerNN, self).__init__()
        non_linearity_class = string2class(non_linearity)
        if h_size == 0:
            self.MLP = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_size, num_classes))
        else:
            self.MLP = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(in_size, h_size), non_linearity_class(), nn.Dropout(dropout),
                                     nn.Linear(h_size, num_classes))

    def forward(self, h):
        return self.MLP(h)