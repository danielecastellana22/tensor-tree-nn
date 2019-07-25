import logging
import os
import torch as th
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

NAME_VAR = 'main'

def set_main_logger_settings(log_dir, name):
    global NAME_VAR

    NAME_VAR = name

    logger = logging.getLogger(NAME_VAR)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")

    # file logger
    fh = logging.FileHandler(os.path.join(log_dir, NAME_VAR) + '.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def get_new_logger(name):
    global NAME_VAR
    logger = logging.getLogger(NAME_VAR+'.'+name)
    #logger.setLevel(logging.DEBUG)
    #formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    return logger


def load_vocabulary(data_dir, logger):
    object_file = os.path.join(data_dir, 'vocab.pkl')
    text_file = os.path.join(data_dir, 'vocab.txt')
    if os.path.exists(object_file):
        # load vocab file
        vocab = th.load(object_file)
    else:
        # create vocab file
        vocab = OrderedDict()
        logger.debug('Loading vocabulary.')
        with open(text_file, encoding='utf-8') as vf:
            for line in tqdm(vf.readlines(), desc='Loading vocabulary: '):
                line = line.strip()
                vocab[line] = len(vocab)
        th.save(vocab, object_file)

    logger.info('Vocabulary loaded.')
    return vocab


def load_embeddings(data_dir, pretrained_emb_file, vocab, logger):
    object_file = os.path.join(data_dir, 'pretrained_emb.pkl')
    embeding_dim = 300
    if os.path.exists(object_file):
        pretrained_emb = th.load(object_file)
    else:
        # filter glove
        glove_emb = {}
        logger.debug('Loading pretrained embeddings.')
        with open(pretrained_emb_file, 'r', encoding='utf-8') as pf:
            for line in tqdm(pf.readlines(), desc='Loading pretrained embeddings:'):
                sp = line.split(' ')
                if sp[0] in vocab:
                    glove_emb[sp[0]] = np.array([float(x) for x in sp[1:]])

        # initialize with glove
        pretrained_emb = np.random.uniform(-0.05, 0.05, (len(vocab), embeding_dim))
        fail_cnt = 0
        for line in vocab.keys():
            if line in glove_emb:
                pretrained_emb[vocab[line], :] = glove_emb[line]
            else:
                fail_cnt += 1

        logger.info('Miss word in GloVe {0:.4f}'.format(1.0 * fail_cnt / len(pretrained_emb)))
        pretrained_emb = th.tensor(pretrained_emb).float()
        th.save(pretrained_emb, object_file)

    logger.info('Pretrained embeddings loaded.')
    return pretrained_emb