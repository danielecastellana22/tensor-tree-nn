import os

import argparse
import torch.nn.init as INIT
import torch.optim as optim

from treeLSTM.utils import set_main_logger_settings
from treeLSTM.cells import *
from treeLSTM.trainer import *
from treeLSTM.metrics import Accuracy

from tests.LRT.utils import load_lrt_dataset, create_lrt_model, lrt_loss_function, lrt_extract_batch_data

def main(args):

    # create log_dir
    log_dir = os.path.join(args.save, args.expname)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # initiliase the main ogger
    main_logger = set_main_logger_settings(log_dir, 'main')

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
    trainset, devset, testset_list = load_lrt_dataset(args.max_n_operator)

    # create the model
    model = create_lrt_model(args.x_size, args.h_size, args.cell_type, args.rank, args.pos_stationarity).to(device)

    params_ex_emb = [x for x in list(model.parameters()) if x.requires_grad]

    for p in params_ex_emb:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    # create the optimizer
    optimizer = optim.Adagrad([
        {'params': params_ex_emb, 'lr': args.lr, 'weight_decay': args.weight_decay, 'lr_decay':0.05}])

    # train and validate
    best_model, best_dev_metrics, *others = train_and_validate(model, lrt_extract_batch_data, lrt_loss_function, optimizer, trainset, devset, device,
                                                      metrics_class=[Accuracy],
                                                      batch_size=args.batch_size,
                                                      n_epochs=args.epochs, early_stopping_patience=args.early_stopping)

    for i,testset in enumerate(testset_list):
        main_logger.info('Test set of len '+str(i))
        test(best_model, lrt_extract_batch_data, testset, device,
             metrics_class=[Accuracy],
             batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #TODO: expanme anch savedit can be decided programmatically
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--cell-type', default='nary')
    parser.add_argument('--max-n-operator', type=int, default=4)
    parser.add_argument('--x-size', type=int, default=75)
    parser.add_argument('--h-size', type=int, default=75)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--save', default='checkpoints/')
    parser.add_argument('--expname', default='lrt-test')
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--pos-stationarity', dest='pos_stationarity', action='store_true')
    args = parser.parse_args()
    #print(args)
    main(args)
