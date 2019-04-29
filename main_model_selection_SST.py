import os
import argparse
import torch as th
import torch.nn.init as INIT
import torch.optim as optim
from treeLSTM import *
from cannon import ParamListTrainer


def get_train_and_validate_fun(args):
    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)
    else:
        th.set_num_threads(10)


    # load the data
    trainset = SSTDataset('data/sst/', 'train.txt', glove300_file='data/glove.840B.300d.txt')
    devset = SSTDataset('data/sst/', 'dev.txt', glove300_file='data/glove.840B.300d.txt')
    #testset = SSTDataset('data/sst/', 'test.txt', glove300_file='data/glove.840B.300d.txt')

    def train_foo(id, log_dir, params):
        set_main_logger_settings(log_dir, 'exp{}'.format(id))
        logger = get_new_logger('main')

        model = TreeLSTM(trainset.num_vocabs,
                         args.x_size,
                         params["h_size"],
                         trainset.num_classes,
                         args.dropout,
                         pretrained_emb=trainset.pretrained_emb,
                         cell_type=args.cell_type).to(device)

        # log model info
        logger.info(str(model))

        params_ex_emb =[x for x in list(model.parameters()) if x.requires_grad and x.size(0)!=trainset.num_vocabs]
        params_emb = list(model.embedding.parameters())

        for p in params_ex_emb:
            if p.dim() > 1:
                INIT.xavier_uniform_(p)

        optimizer = optim.Adagrad([
            {'params': params_ex_emb, 'lr': params['lr'], 'weight_decay': args.weight_decay},
            {'params': params_emb, 'lr': 0.1}])

        best_model, best_dev_metrics = train_and_validate(model, optimizer, trainset, devset, device,
                                                          metrics_class=[LabelAccuracy, RootAccuracy, LeavesAccuracy],
                                                          batch_size=args.batch_size,
                                                          n_epochs=args.epochs,
                                                          early_stopping_patience=args.early_stopping)

        th.save(best_model.state_dict(), os.path.join(log_dir, 'best.pkl'))

        #test on training set
        training_metrics = test(best_model, trainset, device,
                                metrics_class=[LabelAccuracy, RootAccuracy, LeavesAccuracy],
                                batch_size=args.batch_size)
        ris = {}
        ris['tr_loss'] = training_metrics[0].get_value()
        ris['tr_acc'] = training_metrics[1].get_value()
        ris['vl_loss'] = best_dev_metrics[0].get_value()
        ris['vl_acc'] = best_dev_metrics[1].get_value()
        return ris

    return train_foo


if __name__ == '__main__':
    # TODO: add early stopping as parameter
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
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--save', default='checkpoints/')
    parser.add_argument('--expname', default='test')
    args = parser.parse_args()
    #print(args)
    trainer_fun = get_train_and_validate_fun(args)
    #Model selection experiment
    exp_dir = os.path.join(args.save, args.expname)

    lr_list = [0.005, 0.01, 0.02]
    hsize_list = [20, 50, 100, 150]
    param_list = []
    for lr in lr_list:
        for hsize in hsize_list:
            d = {}
            d['lr'] = lr
            d['h_size'] = hsize
            param_list.append(d)

    m_sel = ParamListTrainer(exp_dir, param_list, trainer_fun)
    m_sel.foo()
