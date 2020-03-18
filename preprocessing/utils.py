from tqdm import tqdm
import numpy as np
from utils.utils import eprint

class ConstValues:
    UNK = 0
    NO_ELEMENT = -1


def load_embeddings(pretrained_embs_file, vocab, embedding_dim):

    # filter glove
    glove_emb = {}
    with open(pretrained_embs_file, 'r', encoding='utf-8') as pf:
        for line in tqdm(pf.readlines(), desc='Loading pretrained embeddings:'):
            sp = line.split(' ')
            if sp[0] in vocab:
                glove_emb[sp[0]] = np.array([float(x) for x in sp[1:]])

    # initialize with glove
    pretrained_embs = np.random.uniform(-0.05, 0.05, (len(vocab), embedding_dim))
    fail_cnt = 0
    for line, v in vocab.items():
        if v != ConstValues.UNK:
            if line in glove_emb:
                pretrained_embs[v, :] = glove_emb[line]
            else:
                fail_cnt += 1

    eprint('Missing words in GloVe: {0:.2f}%'.format(100.0 * fail_cnt / len(pretrained_embs)))

    assert pretrained_embs.shape[0] == len(vocab)
    return pretrained_embs


