import os
from tqdm import tqdm
import torch as th
from .utils import get_new_logger
import copy

def train(model, trainset):
    raise Exception('This function is not implemented yet!')


def train_and_validate(model, loss_function, optimizer, trainset, devset, device, metrics_class, batch_size=25, n_epochs=200, early_stopping_patience=20):
    logger = get_new_logger('train_and_validate')

    best_dev_metric = 0
    trainloader = trainset.get_loader(batch_size, device, shuffle=True)
    devloader = devset.get_loader(batch_size, device)

    best_metrics = []
    best_epoch = -1
    best_model = None

    for epoch in range(n_epochs):
        model.train()

        metrics = []
        for c in metrics_class:
            metrics.append(c())

        with tqdm(total=len(trainset), desc='Training epoch ' + str(epoch) + ': ') as pbar:
            for step, batch in enumerate(trainloader):
                g = batch.graph
                n = g.number_of_nodes()
                h = th.zeros((n, model.h_size)).to(device)
                c = th.zeros((n, model.h_size)).to(device)

                model_output = model(batch, h, c)
                loss = loss_function(model_output,batch.label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(g.batch_size)

        # eval on dev set
        model.eval()
        with tqdm(total=len(devset), desc='Validate epoch ' + str(epoch) + ' on dev set: ') as pbar:
            for step, batch in enumerate(devloader):
                g = batch.graph
                n = g.number_of_nodes()
                with th.no_grad():
                    h = th.zeros((n, model.h_size)).to(device)
                    c = th.zeros((n, model.h_size)).to(device)
                    out = model(batch, h, c)

                # update all metrics
                for v in metrics:
                    v.update_metric(out, batch)

                pbar.update(g.batch_size)

        # print metrics
        s = "Dev Test: Epoch {:03d} | ".format(epoch)
        for v in metrics:
            v.finalise_metric()
            s += str(v) + " | "
        logger.info(s)

        # the metrics in poisiton 0 is the one used to validate the model
        if metrics[0].is_better_than(best_dev_metric):
            best_dev_metric = metrics[0].get_value()
            best_epoch = epoch
            best_metrics = metrics
            best_model = copy.deepcopy(model)
            logger.debug('Epoch {:03d}: New optimum found'.format(epoch))
        else:
            # early stopping
            if best_epoch <= epoch - early_stopping_patience:
                break

        # lr decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(1e-5, param_group['lr'] * 0.99)  # 10

    return best_model, best_metrics


def test(model, testset, device, metrics_class, batch_size=25):
    logger = get_new_logger('test')

    testloader = testset.get_loader(batch_size, device)

    test_metrics = []
    for c in metrics_class:
        test_metrics.append(c())

    model.eval()
    with tqdm(total=len(testset), desc='Testing on test set: ') as pbar:
        for step, batch in enumerate(testloader):
            g = batch.graph
            n = g.number_of_nodes()
            with th.no_grad():
                h = th.zeros((n, model.h_size)).to(device)
                c = th.zeros((n, model.h_size)).to(device)
                out = model(batch, h, c)

            # update all metrics
            for v in test_metrics:
                v.update_metric(out, batch)

            pbar.update(g.batch_size)

    # print metrics
    s = "Test: "
    for v in test_metrics:
        v.finalise_metric()
        s += str(v) + " | "
    logger.info(s)

    return test_metrics
