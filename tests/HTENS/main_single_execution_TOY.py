import os

import argparse
import collections
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import logging

from treeLSTM import *


def main(args):

    # create log_dir
    log_dir = os.path.join(args.save, args.expname)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # initiliase the main ogger
    set_main_logger_settings(log_dir, 'main')

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
    trainset = ToyDataset(args.ds_path, 'train.txt')
    devset = ToyDataset(args.ds_path, 'dev.txt')
    testset = ToyDataset(args.ds_path, 'test.txt')

    # create the model
    model = TreeLSTM(2, args.x_size, args.h_size, 2, args.dropout, cell_type=args.cell_type).to(device)

    params_ex_emb = [x for x in list(model.parameters()) if x.requires_grad and x.size(0) != 2]
    #params_emb = list(model.embedding.parameters())

    for p in params_ex_emb:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    # create the optimizer
    optimizer = optim.Adagrad([
        {'params':params_ex_emb, 'lr':args.lr, 'weight_decay':args.weight_decay}])
     #   {'params':params_emb, 'lr':0.1*args.lr}])

    # train and validate
    best_model, best_dev_metrics = train_and_validate(model, optimizer, trainset, devset, device,
                                                      metrics_class=[LabelAccuracy, RootAccuracy, LeavesAccuracy],
                                                      batch_size=args.batch_size,
                                                      n_epochs=args.epochs)

    test(best_model, testset, device,
         metrics_class=[LabelAccuracy, RootAccuracy, LeavesAccuracy],
         batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--ds-path', default='data/htens/')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--cell-type', default='nary')
    parser.add_argument('--x-size', type=int, default=3)
    parser.add_argument('--h-size', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--save', default='checkpoints/')
    parser.add_argument('--expname', default='test')
    args = parser.parse_args()
    #print(args)
    main(args)
