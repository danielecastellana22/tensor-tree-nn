import logging
import os
import torch as th
from tqdm import tqdm
import numpy as np
from datetime import datetime


def import_dataset_utils(dataset_name):
    if dataset_name == 'sst_nary_const' or dataset_name == 'sst_nary_dep':
        from experiments.SST_nary.utils import SST_single_run_fun, get_SST_model_selection_fun
        return SST_single_run_fun, get_SST_model_selection_fun
    elif dataset_name == 'sick':
        from experiments.SICK.utils import SICK_single_run_fun, get_SICK_model_selection_fun
        return SICK_single_run_fun, get_SICK_model_selection_fun
    elif dataset_name == 'lrt':
        from experiments.LRT.utils import LRT_single_run_fun, get_LRT_model_selection_fun
        return LRT_single_run_fun, get_LRT_model_selection_fun
    else:
        raise ValueError('Dataset {} is not known.'.format(dataset_name))


__base_logger_name__ = None


def init_base_logger(log_dir, base_logger_name='main'):
    global __base_logger_name__
    __base_logger_name__ = base_logger_name
    logger = logging.getLogger(base_logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")

    # file logger
    fh = logging.FileHandler(os.path.join(log_dir, base_logger_name) + '.log', mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def get_base_logger():
    global __base_logger_name__
    return logging.getLogger(__base_logger_name__)


def get_sub_logger(sub_name):
    global __base_logger_name__
    return logging.getLogger(__base_logger_name__+'.'+sub_name)


def load_embeddings(data_dir, pretrained_emb_file, vocab):

    logger = get_sub_logger('embeddings')

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


def create_log_dir(par_dir, dataset_name, model_name):

    datetime_dir = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = os.path.join(os.path.join(os.path.join(par_dir, dataset_name), model_name), datetime_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir
