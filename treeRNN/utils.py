import torch as th
import torch.nn as nn


class WordEmbeddingsModule(nn.Module):

    def __init__(self, emb_size, pretrained_embs=None, num_vocabs=None, n_tags=-1, tag_emb_size=50, freeze=False):
        super(WordEmbeddingsModule, self).__init__()

        self.emb_size = emb_size
        self.tag_emb_size = tag_emb_size
        if pretrained_embs is not None:
            self.emb_layer = nn.Embedding.from_pretrained(pretrained_embs, freeze=freeze)
        else:
            self.emb_layer = nn.Embedding(num_vocabs, emb_size)

        self.use_tag = False
        if n_tags != -1:
            self.use_tag = True
            self.tag_emb_layer = nn.Embedding(n_tags, tag_emb_size)

    def forward(self, g):
        n_batch = g.ndata['x'].size(0)
        x = g.ndata['x']
        mask = g.ndata['mask'].bool()
        out = th.zeros((n_batch, self.emb_size), device=x.device)
        out[mask,:] = self.emb_layer(x[mask])

        if self.use_tag:
            tag_id = g.ndata['tag_id']
            tag_embs = th.zeros((n_batch, self.tag_emb_size), device=tag_id.device)
            tag_embs[mask] = self.tag_emb_layer(tag_id[mask])
            out = th.sum(th.bmm(out.view((-1, self.emb_size, 1)), tag_embs.view(-1, 1, self.tag_emb_size)), dim=2)
        return out

