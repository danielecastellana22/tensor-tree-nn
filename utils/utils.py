from pydoc import locate
import numpy as np
import torch as th
import os
import logging
from datetime import datetime
from tqdm import tqdm


def set_initial_seed(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def string2class(string):
    c = locate(string)
    if c is None:
        raise ModuleNotFoundError('{} cannot be found!'.format(string))
    return c


def create_datatime_dir(par_dir):
    datetime_dir = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = os.path.join(par_dir, datetime_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def get_logger(name, log_dir, write_on_console):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")

    # file logger
    fh = logging.FileHandler(os.path.join(log_dir, name) + '.log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if write_on_console:
        # console logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def load_embeddings(data_dir, pretrained_embs_file, vocab, logger, unk_id=0):

    object_file = os.path.join(data_dir, 'pretrained_emb.pkl')

    embedding_dim = 300
    if os.path.exists(object_file):
        pretrained_emb = th.load(object_file)
    else:
        # filter glove
        glove_emb = {}
        logger.debug('Loading pretrained embeddings.')
        with open(pretrained_embs_file, 'r', encoding='utf-8') as pf:
            for line in tqdm(pf.readlines(), desc='Loading pretrained embeddings:'):
                sp = line.split(' ')
                if sp[0] in vocab:
                    glove_emb[sp[0]] = np.array([float(x) for x in sp[1:]])

        # initialize with glove
        pretrained_emb = np.random.uniform(-0.05, 0.05, (len(vocab), embedding_dim))
        fail_cnt = 0
        for line, v in vocab.items():
            if v != unk_id and line in glove_emb:
                pretrained_emb[v, :] = glove_emb[line]
            else:
                fail_cnt += 1

        logger.info('Miss word in GloVe {0:.4f}'.format(1.0 * fail_cnt / len(pretrained_emb)))
        pretrained_emb = th.tensor(pretrained_emb).float()
        th.save(pretrained_emb, object_file)

    logger.info('Pretrained embeddings loaded.')
    assert pretrained_emb.size(0) == len(vocab)
    return pretrained_emb