import os
# to redcue the number of thread in each process
os.environ['OMP_NUM_THREADS'] = '1'
from exputils.configurations import ExpConfig
import argparse
from exputils.runners import ExperimentRunner
from tasks.sick.exp_utils import PearsonSICK

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--config-file', dest='config_file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--num-workers', dest='num_workers', default=10, type=int)
    # TODO: add other args for recovery
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # load the config file
    exp_runner_params, config_list = ExpConfig.from_file(args.config_file)

    # create output dir
    exp_runner = ExperimentRunner(**exp_runner_params, config_list=config_list,
                                  num_workers=args.num_workers, debug_mode=args.debug)
    exp_runner.run()
