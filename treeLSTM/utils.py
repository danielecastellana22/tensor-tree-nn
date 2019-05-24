import logging
import os

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