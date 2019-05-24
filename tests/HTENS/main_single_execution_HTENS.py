import os

import argparse
import numpy as np
import torch as th
import torch.nn.init as INIT
import torch.optim as optim

from treeLSTM import *
from utils import load_htens_dataset, create_htens_model, htens_loss_function, htens_extract_batch_data

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
    trainset, devset, testset = load_htens_dataset()

    # create the model
    model = create_htens_model(args.x_size, args.h_size, args.dropout, cell_type=args.cell_type).to(device)

    params_ex_emb = [x for x in list(model.parameters()) if x.requires_grad]

    for p in params_ex_emb:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    # create the optimizer
    optimizer = optim.Adagrad([
        {'params':params_ex_emb, 'lr':args.lr, 'weight_decay':args.weight_decay}])

    # train and validate
    best_model, best_dev_metrics = train_and_validate(model, htens_extract_batch_data, htens_loss_function, optimizer, trainset, devset, device,
                                                      metrics_class=[Accuracy, RootAccuracy, LeavesAccuracy],
                                                      batch_size=args.batch_size,
                                                      n_epochs=args.epochs, early_stopping_patience=args.early_stopping)

    test(best_model, htens_extract_batch_data, testset, device,
         metrics_class=[Accuracy, RootAccuracy, LeavesAccuracy],
         batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #TODO: expanme anch savedit can be decided programmatically
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--cell-type', default='nary')
    parser.add_argument('--x-size', type=int, default=10)
    parser.add_argument('--h-size', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--save', default='checkpoints/')
    parser.add_argument('--expname', default='test')
    args = parser.parse_args()
    #print(args)
    main(args)
