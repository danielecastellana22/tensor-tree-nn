import os
# to redcue the number of thread in each process
os.environ['OMP_NUM_THREADS'] = '1'
from config.base import ExpConfig
import argparse
from experiments.runner import ExperimentRunner


def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--config-file', dest='config_file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # load the config file
    exp_runner_params, config_list = ExpConfig.from_file(args.config_file)

    # create output dir
    exp_runner = ExperimentRunner(**exp_runner_params, config_list=config_list)
    exp_runner.run()
