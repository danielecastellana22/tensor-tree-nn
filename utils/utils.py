import sys
import os
from pydoc import locate
import numpy as np
import torch as th
import logging
from datetime import datetime


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
