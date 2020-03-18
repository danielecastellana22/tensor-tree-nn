import sys
import os
import argparse
from utils.config import Config
from utils.utils import eprint


def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--config-file', dest='config_file')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    # load the config file
    c = Config.from_file(args.config_file)
    output_dir = c.output_dir

    if not os.path.exists(output_dir):
        eprint('Create otutput dir')
        os.makedirs(output_dir)
    else:
        eprint('Output dir already exists! Content will be overwritten! Continue?')
        sys.stdin.readline()

    eprint("Starts preprocessing function!")

    p_obj = c.preprocessor_class(c)
    p_obj.preprocess()