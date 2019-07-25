import os

import argparse
import numpy as np
import torch as th
import torch.nn.init as INIT
import torch.optim as optim

from treeLSTM.utils import set_main_logger_settings, load_embeddings, load_vocabulary
from treeLSTM.trainer import train_and_validate, test
from treeLSTM.metrics import Accuracy, LeavesAccuracy, RootAccuracy

from tests.SST_dep_nary.utils import create_model, load_dataset, loss_function, extract_batch_data, MaskedAccuracy


def main(args):

    # create log_dir
    log_dir = os.path.join(args.save, args.expname)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # initiliase the main ogger
    logger = set_main_logger_settings(log_dir, 'main')

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
        th.set_num_threads(10)

    # load the data
    trainset, devset, testset = load_dataset()

    pretrained_embs = load_embeddings('data/sst_nary/dep_tree/',
                                      pretrained_emb_file='data/glove.840B.300d.txt',
                                      vocab=trainset.get_vocab(), logger=logger)


    # create the model
    model = create_model(args.x_size, args.h_size, trainset.num_classes,
                         max_output_degree=trainset.max_out_degree,
                         dropout=args.dropout,
                         pretrained_emb=pretrained_embs,
                         cell_type=args.cell_type,
                         rank=args.rank,
                         pos_stationarity=args.pos_stationarity).to(device)

    params_ex_emb = [x for x in list(model.parameters()) if x.requires_grad and x.size(0) != trainset.num_vocabs]
    params_emb = list(model.input_module.parameters())

    for p in params_ex_emb:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    # create the optimizer
    optimizer = optim.Adagrad([
        {'params': params_ex_emb, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': params_emb, 'lr': 0.1*args.lr}])

    # train and validate
    best_model, best_dev_metrics, *others = train_and_validate(model, extract_batch_data, loss_function, optimizer, trainset, devset, device,
                                                      metrics_class=[MaskedAccuracy, RootAccuracy, LeavesAccuracy],
                                                      batch_size=args.batch_size,
                                                      n_epochs=args.epochs, early_stopping_patience=args.early_stopping)

    test(best_model, extract_batch_data,  testset, device,
         metrics_class=[MaskedAccuracy, RootAccuracy, LeavesAccuracy],
         batch_size=args.batch_size)


if __name__ == '__main__':
    #TODO: expanme anch savedit can be decided programmatically
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--cell-type', default='nary')
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=160)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--pos-stationarity', dest='pos_stationarity', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--save', default='checkpoints/')
    parser.add_argument('--expname', default='test')
    args = parser.parse_args()
    #print(args)
    main(args)
