import argparse
import json
import numpy as np
import torch as th
from experiments.execution_utils import init_base_logger, get_base_logger, create_log_dir, import_dataset_utils


def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--save-dir', default='checkpoints/')
    parser.add_argument('--dataset')
    parser.add_argument('--tree-model')
    parser.add_argument('--cell-type')
    # learning params
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.05)
    # tree-LSTM params
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=160)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--pos-stationarity', dest='pos_stationarity', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    #parser.add_argument('--cell-weight-decay', type=float, default=1e-4)
    # other params in JSON format that depend on the experiment
    parser.add_argument('--others', default=None)

    args = parser.parse_args()

    # add the others params to the args object
    if args.others is not None:
        other_param_dict = json.loads(args.others.replace("'", '"'))
        for k in other_param_dict:
            setattr(args, k, other_param_dict[k])

    return args


if __name__ == '__main__':

    args = parse_arguments()

    # create log_dir
    log_dir = create_log_dir(args.save_dir, args.dataset, args.cell_type)

    # initiliase the main ogger
    init_base_logger(log_dir, 'main')
    logger = get_base_logger()
    logger.info(str(args))

    # set the seed
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    # set the device
    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)
    else:
        th.set_num_threads(-args.gpu)

    single_run_fun, _ = import_dataset_utils(args.dataset)

    single_run_fun(args, device, log_dir)
