from tqdm import tqdm
import torch as th
import copy
from .metrics import ValueMetric, TreeMetric
import time
from torch.utils.data import DataLoader


def __evaluate_model__(model, dataloader, metric_class_list, pbar, batch_size):
    predictions = []
    eval_time = 0
    metrics = []
    for c in metric_class_list:
        metrics.append(c())

    model.eval()
    for step, batch in enumerate(dataloader):

        t = time.time()
        in_data = batch[0]
        out_data = batch[1]
        with th.no_grad():
            out = model(*in_data)

        predictions.append(out)

        # update all metrics
        for v in metrics:
            if isinstance(v, ValueMetric):
                v.update_metric(out, out_data)

            if isinstance(v, TreeMetric):
                v.update_metric(out, out_data, *in_data)
        eval_time += (time.time() - t)

        pbar.update(min(batch_size, pbar.total - pbar.n))

    pbar.close()

    return metrics, eval_time, predictions


def train_and_validate(model, loss_function, optimizer, trainset, valset, batcher_fun, list_metrics_class, logger,
                       batch_size, n_epochs, early_stopping_patience, evaluate_on_training_set):

    train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=batcher_fun, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=batch_size, collate_fn=batcher_fun, shuffle=True, num_workers=0)

    best_dev_metric = None
    best_epoch = -1
    best_model = None

    dev_metrics = {}
    tr_metrics = {}
    for c in list_metrics_class:
        dev_metrics[c.__name__] = []
        tr_metrics[c.__name__] = []

    tr_forw_time_list = []
    tr_backw_time_list = []
    dev_val_time_list = []

    for epoch in range(1, n_epochs+1):
        model.train()

        tr_forw_time = 0
        tr_backw_time = 0

        with tqdm(total=len(trainset), desc='Training epoch ' + str(epoch) + ': ') as pbar:
            for step, batch in enumerate(train_loader):

                t = time.time()
                in_data = batch[0]
                out_data = batch[1]
                model_output = model(*in_data)
                loss = loss_function(model_output, out_data)
                tr_forw_time += (time.time() - t)

                t = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tr_backw_time += (time.time() - t)

                pbar.update(min(batch_size, pbar.total-pbar.n))

        if evaluate_on_training_set:
            # eval on tr set
            pbar = tqdm(total=len(trainset), desc='Evaluate epoch ' + str(epoch) + ' on training set: ')
            metrics, _, _ = __evaluate_model__(model, train_loader, list_metrics_class, pbar, batch_size)

            # print tr metrics
            s = "Evaluation on training set: Epoch {:03d} | ".format(epoch)
            for v in metrics:
                v.finalise_metric()
                s += str(v) + " | "
                tr_metrics[type(v).__name__].append(v.get_value())
            logger.info(s)

        # eval on dev set
        pbar = tqdm(total=len(valset), desc='Evaluate epoch ' + str(epoch) + ' on dev set: ')
        metrics, eval_dev_time, _ = __evaluate_model__(model, val_loader, list_metrics_class, pbar, batch_size)

        # print dev metrics
        s = "Evaluation on dev set: Epoch {:03d} | ".format(epoch)
        for v in metrics:
            v.finalise_metric()
            s += str(v) + " | "
            dev_metrics[type(v).__name__].append(v.get_value())
        logger.info(s)

        # early stopping
        if best_dev_metric is None:
            best_dev_metric = copy.deepcopy(metrics[0])
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        else:
            # the metrics in poisiton 0 is the one used to validate the model
            if metrics[0].is_better_than(best_dev_metric):
                best_dev_metric = copy.deepcopy(metrics[0])
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                logger.info('Epoch {:03d}: New optimum found'.format(epoch))
            else:
                # early stopping
                if best_epoch <= epoch - early_stopping_patience:
                    break

        tr_forw_time_list.append(tr_forw_time)
        tr_backw_time_list.append(tr_backw_time)
        dev_val_time_list.append(eval_dev_time)

    # build vocabulary for the result
    info_training = {
        'best_epoch': best_epoch,
        'tr_metrics': tr_metrics,
        'dev_metrics': dev_metrics,
        'tr_forward_time': tr_forw_time_list,
        'tr_bakcward_time': tr_backw_time_list,
        'dev_eval_time': dev_val_time_list}

    return best_dev_metric, best_model, info_training


def test(model, testset, batcher_fun, metric_class_list, logger, batch_size):

    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=batcher_fun, shuffle=True, num_workers=0)

    pbar = tqdm(total=len(testset), desc='Evaluate on test set: ')
    metrics, eval_dev_time, predictions = __evaluate_model__(model, testloader, metric_class_list, pbar, batch_size)

    test_metrics = {}
    # print metrics
    s = "Test: "
    for v in metrics:
        v.finalise_metric()
        s += str(v) + " | "
        test_metrics[type(v).__name__] = v.get_value()

    logger.info(s)

    return test_metrics, predictions
