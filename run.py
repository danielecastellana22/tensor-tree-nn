from utils.config import ExpConfig
import argparse
from utils.experiment import ExperimentRunner
from utils.utils import create_datatime_dir


def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--config-file', dest='config_file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # load the config file
    exp_config, config_list = ExpConfig.from_file(args.config_file)

    # create output dir
    exp_runner = ExperimentRunner(**exp_config, config_list=config_list)
    exp_runner.run()
