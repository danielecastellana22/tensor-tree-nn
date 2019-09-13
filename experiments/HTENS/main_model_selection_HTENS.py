import os
import argparse
import torch as th
import torch.nn.init as INIT
import torch.optim as optim
from cannon import ParamListTrainer
from experiments.execution_utils import set_main_logger_settings, get_new_logger
from treeLSTM.trainer import train_and_validate, test
from treeLSTM.metrics import Accuracy, RootAccuracy, LeavesAccuracy
from experiments.HTENS.utils import load_htens_dataset, create_htens_model, htens_loss_function, htens_extract_batch_data


def get_train_and_validate_fun(args):
    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)
    else:
        th.set_num_threads(10)

    # load the data
    trainset, devset, testset = load_htens_dataset(args.data_dir)


    def train_foo(id, log_dir, params):
        set_main_logger_settings(log_dir, 'exp{}'.format(id))
        logger = get_new_logger('main')

        # create the model
        model = create_htens_model(args.x_size,  params['h_size'], args.dropout, max_output_degree=trainset.max_out_degree,
                                   cell_type=args.cell_type, rank=params['rank'], pos_stationarity=args.pos_stationarity).to(device)

        # log model info
        logger.info(str(model))

        params_ex_emb =[x for x in list(model.parameters()) if x.requires_grad]

        for p in params_ex_emb:
            if p.dim() > 1:
                INIT.xavier_uniform_(p)

        optimizer = optim.Adagrad([
            {'params': params_ex_emb, 'lr': params['lr'], 'weight_decay': args.weight_decay}])

        best_model, best_dev_metrics, best_epoch, tr_forw_time, tr_backw_time, tr_val_time = train_and_validate(model,
                                                                                                             htens_extract_batch_data,
                                                                                                             htens_loss_function,
                                                                                                             optimizer, trainset, devset, device,
                                                                                                             metrics_class=[Accuracy, RootAccuracy, LeavesAccuracy],
                                                                                                             batch_size=args.batch_size,
                                                                                                             n_epochs=args.epochs,
                                                                                                             early_stopping_patience=args.early_stopping)

        #th.save(best_model.state_dict(), os.path.join(log_dir, 'best.pkl'))

        #test on training set
        training_metrics = test(best_model, htens_extract_batch_data, trainset, device,
                                metrics_class=[Accuracy, RootAccuracy, LeavesAccuracy],
                                batch_size=args.batch_size)

        ris = {}
        ris['nodes_tr'] = training_metrics[0].get_value()
        ris['root_tr'] = training_metrics[1].get_value()
        ris['nodes_val'] = best_dev_metrics[0].get_value()
        ris['root_val'] = best_dev_metrics[1].get_value()
        ris['best_epoch'] = best_epoch
        ris['n_cell_param'] = sum(p.numel() for p in model.cell.parameters() if p.requires_grad)
        ris['tr_forw_time'] = sum(tr_forw_time)
        ris['tr_backw_time'] = sum(tr_backw_time)
        ris['val_time'] = sum(tr_val_time)
        return ris

    return train_foo


if __name__ == '__main__':
    #TODO: expanme anch savedit can be decided programmatically
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--cell-type', default='nary')
    parser.add_argument('--x-size', type=int, default=10)
    #parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--log-every', type=int, default=5)
    #parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--save', default='checkpoints/')
    parser.add_argument('--expname', default='test')
    parser.add_argument('--data-dir', default='data/htens')
    parser.add_argument('--rank', type=int, default=20)

    args = parser.parse_args()
    #print(args)
    trainer_fun = get_train_and_validate_fun(args)
    #Model selection experiment
    exp_dir = os.path.join(args.save, args.expname)

    lr_list = [0.01, 0.02, 0.05]
    if args.data_dir == 'data/htens':
        h_to_rank = lambda x: -1
        args.pos_stationarity = False
        if args.cell_type == 'nary':
            hsize_list = [9, 22, 58, 222, 621]
        elif args.cell_type == 'full':
            hsize_list = [5, 10, 20, 50, 100]
        else:
            raise ValueError('Cell type not supported yet')
    elif args.data_dir == 'data/prod':
        args.pos_stationarity = True
        hsize_list = [5, 10, 20, 50, 100]
        if args.cell_type == 'nary':
            h_to_rank = lambda x: -1
        elif args.cell_type == 'cancomp':
            h_to_rank = lambda x: x//2
        elif args.cell_type == 'hosvd':
            rank_list = [2,3,4,5,5]
            h_to_rank = lambda x: rank_list[hsize_list.index(x)]
        else:
            raise ValueError('Cell type not supported yet')
    else:
        raise ValueError('Data not supported yet')




    it_list = list(range(1, 6))
    param_list = []
    for lr in lr_list:
        for hsize in hsize_list:
            for it in it_list:
                d = {}
                d['lr'] = lr
                d['h_size'] = hsize
                d['rank'] = h_to_rank(hsize)
                d['it'] = it
                param_list.append(d)

    m_sel = ParamListTrainer(exp_dir, param_list, trainer_fun)
    m_sel.foo()
