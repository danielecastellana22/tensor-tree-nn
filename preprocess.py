import os
import argparse
from experiments.config import Config
from utils.misc import eprint, string2class


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    # load the config file
    c = Config.from_yaml_file(args.config_file)
    output_dir = c.output_dir

    if not False: # path_exists_with_message(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        eprint("Starts preprocessing function!")

        prep_class = string2class(c.preprocessor_class)
        p_obj = prep_class(c)
        p_obj.preprocess()