import torch as th
import torch.nn as nn
from utils.serialization import from_pkl_file


class EmbeddingModule(nn.Module):

    def __new__(cls, *args, **kwargs):
        embedding_type = kwargs['embedding_type']
        if embedding_type == 'pretrained':
            np_array = from_pkl_file(kwargs['pretrained_embs'])
            return nn.Embedding.from_pretrained(th.tensor(np_array, dtype=th.float), freeze=kwargs['freeze'])
        elif embedding_type == 'one_hot':
            num_embs = kwargs['num_embs']
            return nn.Embedding.from_pretrained(th.eye(num_embs, num_embs), freeze=kwargs['freeze'])
        elif embedding_type == 'random':
            num_embs = kwargs['num_embs']
            emb_size = kwargs['emb_size']
            return nn.Embedding(num_embs, emb_size)
        else:
            raise ValueError('Embedding type is unkown!')
