import os
import argparse
import torch as th
import torch.nn.init as INIT
import torch.optim as optim

from cannon import ParamListTrainer

from treeLSTM.utils import set_main_logger_settings, get_new_logger, load_vocabulary, load_embeddings
from treeLSTM.trainer import train_and_validate, test

from tests.SICK.utils import create_sick_model, load_sick_dataset, sick_loss_function, sick_extract_batch_data, MSE_sick, Pearson_sick


# TODO: my logger and cannon logger should be unified
def get_train_and_validate_fun(args, logger):
    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)
    else:
        th.set_num_threads(10)

    vocab = load_vocabulary('data/sick/', logger=logger)
    pretrained_embs = load_embeddings('data/sick/', pretrained_emb_file='data/glove.840B.300d.txt', vocab=vocab, logger=logger)
    # load the data
    trainset, devset, testset = load_sick_dataset(vocab)

    def train_foo(id, log_dir, params):
        set_main_logger_settings(log_dir, 'exp{}'.format(id))
        logger = get_new_logger('main')

        # create the model
        model = create_sick_model(args.x_size,
                                  args.h_size,
                                  pretrained_emb=pretrained_embs,
                                  cell_type=args.cell_type, max_output_degree=trainset.max_out_degree, rank=params['rank'],
                                  pos_stationarity=args.pos_stationarity).to(device)

        # log model info
        logger.info(str(params))

        params_ex_emb = [x for x in list(model.parameters()) if x.requires_grad]

        for p in params_ex_emb:
            if p.dim() > 1:
                INIT.xavier_uniform_(p)

        # create the optimizer
        optimizer = optim.Adagrad([{'params': params_ex_emb, 'lr': params['lr'], 'weight_decay': params['weight_decay']}])

        # train and validate
        best_model, best_dev_metrics, *others = train_and_validate(model, sick_extract_batch_data, sick_loss_function, optimizer, trainset, devset, device,
                                                      metrics_class=[MSE_sick, Pearson_sick],
                                                      batch_size=args.batch_size,
                                                      n_epochs=args.epochs, early_stopping_patience=args.early_stopping)

        th.save(best_model.state_dict(), os.path.join(log_dir, 'best.pkl'))

        #test on training set
        training_metrics = test(best_model, sick_extract_batch_data,  trainset, device,
                                metrics_class=[MSE_sick, Pearson_sick],
                                batch_size=args.batch_size)
        ris = {}
        ris['MSE_tr'] = training_metrics[0].get_value()
        ris['Pearson_tr'] = training_metrics[1].get_value()
        ris['MSE_val'] = best_dev_metrics[0].get_value()
        ris['Pearson_val'] = best_dev_metrics[1].get_value()
        return ris

    return train_foo


if __name__ == '__main__':
    #TODO: expanme anch savedit can be decided programmatically
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=89)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--cell-type', default='nary')
    #parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early-stopping', type=int, default=5)
    #parser.add_argument('--lr', type=float, default=0.05)
    #parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--pos-stationarity', dest='pos_stationarity', action='store_true')
    parser.add_argument('--save', default='checkpoints/')
    parser.add_argument('--expname', default='test')
    parser.set_defaults(pos_stationarity=False)
    args = parser.parse_args()
    #print(args)

    #Model selection experiment

    # initiliase the main ogger
    logger = set_main_logger_settings(args.save, args.expname)

    trainer_fun = get_train_and_validate_fun(args, logger)

    if args.cell_type == 'cancomp':
        lr_list = [0.01, 0.02, 0.05]
        rank_list = [70, 100, 150]
        weight_decay_list = [1e-5, 1e-4, 1e-3]
        it_list = list(range(1, 6))

        param_list = []
        for lr in lr_list:
            for rank in rank_list:
                for w in weight_decay_list:
                    for it in it_list:
                        d = {}
                        d['lr'] = lr
                        d['it'] = it
                        d['rank'] = rank
                        d['weight_decay'] = w
                        param_list.append(d)
    elif args.cell_type == 'hosvd':
        lr_list = [0.01, 0.02, 0.05]
        rank_list = [2, 3]
        weight_decay_list = [1e-5, 1e-4, 1e-3]
        it_list = list(range(1, 6))

        param_list = []
        for lr in lr_list:
            for rank in rank_list:
                for w in weight_decay_list:
                    for it in it_list:
                        d = {}
                        d['lr'] = lr
                        d['it'] = it
                        d['rank'] = rank
                        d['weight_decay'] = w
                        param_list.append(d)
    elif args.cell_type == 'nary':
        lr_list = [0.02, 0.05]
        weight_decay_list = [1e-4]
        it_list = list(range(1, 6))
        rank_list = [-1]

        param_list = []
        for lr in lr_list:
            for rank in rank_list:
                for w in weight_decay_list:
                    for it in it_list:
                        d = {}
                        d['lr'] = lr
                        d['it'] = it
                        d['rank'] = rank
                        d['weight_decay'] = w
                        param_list.append(d)

    exp_dir = os.path.join(args.save, args.expname)
    m_sel = ParamListTrainer(exp_dir, param_list, trainer_fun)
    m_sel.foo()
