import os
import argparse
from exputils.configurations import Config, create_object_from_config
from exputils.utils import eprint


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    # load the config file
    c = Config.from_yaml_file(args.config_file)
    output_dir = c.output_dir
    input_dir = c.input_dir

    os.makedirs(output_dir, exist_ok=True)
    eprint("Starts preprocessing function!")

    p_obj = create_object_from_config(c.preprocessor, input_dir=input_dir, output_dir=output_dir)
    p_obj.preprocess()
