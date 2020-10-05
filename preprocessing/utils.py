from tqdm import tqdm
import numpy as np
from utils.misc import eprint
import re


class ConstValues:
    UNK = 0
    NO_ELEMENT = -1


def load_embeddings(pretrained_embs_file, vocab, embedding_dim):
    splitting_char = ['\\', '/', '\\/', '-', '_']
    re_split = '|'.join(splitting_char)
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
            s_line = re.split(re_split, line)
            ris = None
            n = 0
            for ll in s_line:
                if ll in glove_emb:
                    if ris is None:
                        ris = np.zeros_like(pretrained_embs[v, :])
                    ris += glove_emb[ll]
                    n += 1
            if ris is not None:
                pretrained_embs[v, :] = ris/n
            else:
                #eprint(line)
                fail_cnt += 1

    eprint('Missing words in GloVe: {} on {} total words'.format(fail_cnt, len(vocab)))

    assert pretrained_embs.shape[0] == len(vocab)
    return pretrained_embs
