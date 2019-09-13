import argparse
import json
from cannon import ParamListTrainer
import torch as th
from experiments.execution_utils import import_dataset_utils, create_log_dir


def build_parameter_list(param_dict):

    para_list = []
    key_list = list(param_dict.keys())

    def rec_build(i, d):
        if i == len(key_list):
            para_list.append(d.copy())
            return

        key = key_list[i]
        for v in param_dict[key]:
            d[key] = v
            rec_build(i+1, d)

    rec_build(0, {})

    return para_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--save-dir', default='checkpoints/')
    parser.add_argument('--dataset')

    parser.add_argument('--cell-type')
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early-stopping', type=int, default=10)

    # this parameter are received in a json format
    parser.add_argument('--parameter-grid')
    parser.add_argument('--n-runs', type=int, default=3)

    parser.add_argument('--restore-dir', default=None)
    parser.add_argument('--message', default='Model selection')
    # other params in JSON format that depend on the experiment
    parser.add_argument('--others', default=None)

    args = parser.parse_args()

    # add the others params to the args object
    if args.others is not None:
        other_param_dict = json.loads(args.others.replace("'", '"'))
        for k in other_param_dict:
            setattr(args, k, other_param_dict[k])

    if args.restore_dir is not None:
        log_dir = args.restore_dir
    else:
        # start new model selection
        log_dir = create_log_dir(args.save_dir, args.dataset, args.cell_type)

    param_dict = json.loads(args.parameter_grid.replace("'", '"'))
    # add the multiple run
    param_dict['run'] = range(args.n_runs)
    param_grid = build_parameter_list(param_dict)

    _, get_ms_trainer_fun = import_dataset_utils(args.dataset)

    # set the device
    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)
    else:
        th.set_num_threads(-args.gpu)

    trainer_fun = get_ms_trainer_fun(args, device)

    m_sel = ParamListTrainer(log_dir, param_grid, trainer_fun)
    m_sel.experiment_log.info(args.message)
    m_sel.experiment_log.info(args)

    m_sel.foo()
