import os
import argparse
import torch as th
import torch.nn.init as INIT
import torch.optim as optim

from cannon import ParamListTrainer

from experiments.execution_utils import set_main_logger_settings, get_new_logger, load_vocabulary, load_embeddings
from treeLSTM.trainer import train_and_validate, test
from treeLSTM.metrics import Accuracy, LeavesAccuracy, RootAccuracy

from experiments.SST.utils import create_sst_model, load_sst_dataset, sst_loss_function, sst_extract_batch_data


def get_train_and_validate_fun(args, logger):
    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)
    else:
        th.set_num_threads(10)

    vocab = load_vocabulary('data/sst/', logger=logger)
    pretrained_embs = load_embeddings('data/sst/', pretrained_emb_file='data/glove.840B.300d.txt', vocab=vocab,
                                     logger=logger)
    # load the data
    trainset, devset, testset = load_sst_dataset(vocab)

    def train_foo(id, log_dir, params):
        set_main_logger_settings(log_dir, 'exp{}'.format(id))
        logger = get_new_logger('main')

        #create the model
        model = create_sst_model(args.x_size, params["h_size"], trainset.num_classes,
                                 max_output_degree=trainset.max_out_degree,
                                 dropout=args.dropout,
                                 pretrained_emb=pretrained_embs,
                                 cell_type=args.cell_type,
                                 rank=params['rank'],
                                 pos_stationarity=args.pos_stationarity).to(device)

        # log model info
        logger.info(str(model))

        params_ex_emb =[x for x in list(model.parameters()) if x.requires_grad and x.size(0)!=trainset.num_vocabs]
        params_emb = list(model.input_module.parameters())

        for p in params_ex_emb:
            if p.dim() > 1:
                INIT.xavier_uniform_(p)

        optimizer = optim.Adagrad([
            {'params': params_ex_emb, 'lr': params['lr'], 'weight_decay':  params['wd']},
            {'params': params_emb, 'lr': 0.1}])

        best_model, best_dev_metrics = train_and_validate(model, sst_extract_batch_data,  sst_loss_function, optimizer, trainset, devset, device,
                                                          metrics_class=[Accuracy, RootAccuracy, LeavesAccuracy],
                                                          batch_size=args.batch_size,
                                                          n_epochs=args.epochs,
                                                          early_stopping_patience=args.early_stopping)

        th.save(best_model.state_dict(), os.path.join(log_dir, 'best.pkl'))

        #test on training set
        training_metrics = test(best_model, sst_extract_batch_data, trainset, device,
                                metrics_class=[Accuracy, RootAccuracy, LeavesAccuracy],
                                batch_size=args.batch_size)
        ris = {}
        ris['tr_node_acc'] = training_metrics[0].get_value()
        ris['tr_root_acc'] = training_metrics[1].get_value()
        ris['vl_node_acc'] = best_dev_metrics[0].get_value()
        ris['vl_root_acc'] = best_dev_metrics[1].get_value()
        return ris

    return train_foo


if __name__ == '__main__':
    #TODO: expanme anch savedit can be decided programmatically
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--cell-type', default='nary')
    parser.add_argument('--x-size', type=int, default=300)
    #parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--log-every', type=int, default=5)
    #parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--pos-stationarity', dest='pos_stationarity', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--save', default='checkpoints/')
    parser.add_argument('--expname', default='test')
    args = parser.parse_args()
    #print(args)
    logger = set_main_logger_settings(args.save, args.expname)
    trainer_fun = get_train_and_validate_fun(args, logger)
    #Model selection experiment
    exp_dir = os.path.join(args.save, args.expname)

    wd_list = [1e-2]#, 1e-3, 1e-2]
    lr_list = [0.01, 0.02, 0.05]
    if args.cell_type == 'nary':
        hsize_list = [10, 25, 66, 255, 714, 937, 1308, 2010]
    else:
        #hsize_list = [5, 10, 20, 50, 100, 120, 150, 200]
        hsize_list = [50, 100, 120]

    it_list = list(range(1, 6))
    param_list = []
    for lr in lr_list:
        for hsize in hsize_list:
            for wd in wd_list:
                for it in it_list:
                    d = {}
                    d['lr'] = lr
                    d['h_size'] = hsize
                    d['wd'] = wd
                    d['it'] = it
                    param_list.append(d)

    m_sel = ParamListTrainer(exp_dir, param_list, trainer_fun)
    m_sel.foo()
