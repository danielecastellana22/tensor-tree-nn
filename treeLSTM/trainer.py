import os
from tqdm import tqdm
import torch as th
import torch.nn.functional as F
from .utils import get_logger
import numpy as np


def train(model, trainset):
    raise Exception('This function is not implemented yet!')


def train_and_validate(model, optimizer, trainset, devset, log_dir, device, n_epochs=200, early_stopping_patience=20, metrics=None):
    logger = get_logger('train_and_validate', log_dir)

    best_epoch = -1
    best_dev_metric = 0
    for epoch in range(n_epochs):
        model.train()

        with tqdm(total=len(trainset), desc='Training epoch ' + str(epoch) + ': ') as pbar:
            for step, batch in enumerate(trainset.get_loader()):
                g = batch.graph
                n = g.number_of_nodes()
                h = th.zeros((n, model.h_size)).to(device)
                c = th.zeros((n, model.h_size)).to(device)

                logits = model(batch, h, c)
                logp = F.log_softmax(logits, 1)
                loss = F.nll_loss(logp, batch.label, reduction='sum')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(g.batch_size)

        # eval on dev set
        dev_accs = []
        dev_root_accs = []
        model.eval()
        with tqdm(total=len(devset), desc='Validate epoch ' + str(epoch) + ' on dev set: ') as pbar:
            for step, batch in enumerate(devset.get_loader()):
                g = batch.graph
                n = g.number_of_nodes()
                with th.no_grad():
                    h = th.zeros((n, model.h_size)).to(device)
                    c = th.zeros((n, model.h_size)).to(device)
                    out = model(batch, h, c)

                #root_ids = [i for i in range(batch.graph.number_of_nodes()) if batch.graph.out_degree(i) == 0]
                #leaves_ids = [i for i in range(batch.graph.number_of_nodes()) if batch.graph.in_degree(i) == 0]
                #pred = th.argmax(logits, 1)
                #acc = th.sum(th.eq(batch.label, pred)).item()
                #dev_accs.append([acc, len(batch.label)])
                #root_acc = np.sum(batch.label.cpu().data.numpy()[root_ids] == pred.cpu().data.numpy()[root_ids])
                #dev_root_accs.append([root_acc, len(root_ids)])
                for (k, v) in metrics.items():
                    v.update_metric(out, batch)

                pbar.update(g.batch_size)



        #dev_acc = 1.0 * np.sum([x[0] for x in dev_accs]) / np.sum([x[1] for x in dev_accs])
        #dev_root_acc = 1.0 * np.sum([x[0] for x in dev_root_accs]) / np.sum([x[1] for x in dev_root_accs])
        s = "Dev Test: Epoch {:03d} | ".format(epoch)
        for (k, v) in metrics.items():
            v.finalise_metric()
            s += "{} {:.4f} | ".format(k, v.get_value())
        logger.info(s)

        # the metrics in poisiton 0 is the one used to validate the model
        if metrics[0].is_better_than(best_dev_metric):
            best_dev_metric = metrics[0].get_value()
            best_epoch = epoch
            th.save(model.state_dict(), os.path.join(log_dir, 'best.pkl'))
            logger.debug('Epoch {:03d}: New optimum found'.format(epoch))
        else:
            # early stopping
            if best_epoch <= epoch - early_stopping_patience:
                break

        # lr decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(1e-5, param_group['lr'] * 0.99)  # 10


def test(model, testset):
    raise Exception('This function is not implemented yet!')
