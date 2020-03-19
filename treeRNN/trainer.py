from tqdm import tqdm
import torch as th
import copy
from .metrics import ValueMetricUpdate, TreeMetricUpdate
import time
from torch.utils.data import DataLoader


def train_and_validate(model, loss_function, optimizer, trainset, valset, batcher_fun, metric_class_list, logger,
                       batch_size, n_epochs, early_stopping_patience, evaluate_on_training_set):

    train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=batcher_fun, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=batch_size, collate_fn=batcher_fun, shuffle=True, num_workers=0)

    best_val_metrics = None
    best_epoch = -1
    best_model = None

    val_metrics = {}
    tr_metrics = {}
    for c in metric_class_list:
        val_metrics[c.__name__] = []
        tr_metrics[c.__name__] = []

    tr_forw_time_list = []
    tr_backw_time_list = []
    val_time_list = []

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
            metrics, _, _ = __evaluate_model__(model, train_loader, metric_class_list, pbar, batch_size)

            # print tr metrics
            s = "Evaluation on training set: Epoch {:03d} | ".format(epoch)
            for v in metrics:
                s += str(v) + " | "
                tr_metrics[type(v).__name__].append(v.get_value())
            logger.info(s)

        # eval on validation set
        pbar = tqdm(total=len(valset), desc='Evaluate epoch ' + str(epoch) + ' on validation set: ')
        metrics, eval_val_time, _ = __evaluate_model__(model, val_loader, metric_class_list, pbar, batch_size)

        # print validation metrics
        s = "Evaluation on validation set: Epoch {:03d} | ".format(epoch)
        for v in metrics:
            s += str(v) + " | "
            val_metrics[type(v).__name__].append(v.get_value())
        logger.info(s)

        # early stopping
        if best_val_metrics is None:
            best_val_metrics = copy.deepcopy(metrics)
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        else:
            # the metrics in poisiton 0 is the one used to validate the model
            if metrics[0].is_better_than(best_val_metrics[0]):
                best_val_metrics = copy.deepcopy(metrics)
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                logger.info('Epoch {:03d}: New optimum found'.format(epoch))
            else:
                # early stopping
                if best_epoch <= epoch - early_stopping_patience:
                    break

        tr_forw_time_list.append(tr_forw_time)
        tr_backw_time_list.append(tr_backw_time)
        val_time_list.append(eval_val_time)

    # print best results
    s = "Best found in Epoch {:03d} | ".format(best_epoch)
    for v in best_val_metrics:
        s += str(v) + " | "
    logger.info(s)

    # build vocabulary for the result
    info_training = {
        'best_epoch': best_epoch,
        'tr_metrics': tr_metrics,
        'val_metrics': val_metrics,
        'tr_forward_time': tr_forw_time_list,
        'tr_bakcward_time': tr_backw_time_list,
        'val_eval_time': val_time_list}

    return best_val_metrics, best_model, info_training


def test(model, testset, batcher_fun, metric_class_list, logger, batch_size):

    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=batcher_fun, shuffle=True, num_workers=0)

    pbar = tqdm(total=len(testset), desc='Evaluate on test set: ')
    metrics, _, predictions = __evaluate_model__(model, testloader, metric_class_list, pbar, batch_size)

    # print metrics
    s = "Test: "
    for v in metrics:
        s += str(v) + " | "

    logger.info(s)

    return metrics, predictions


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
            if isinstance(v, ValueMetricUpdate):
                v.update_metric(out, out_data)

            if isinstance(v, TreeMetricUpdate):
                v.update_metric(out, out_data, *in_data)
        eval_time += (time.time() - t)

        pbar.update(min(batch_size, pbar.total - pbar.n))

    pbar.close()

    for v in metrics:
        v.finalise_metric()

    return metrics, eval_time, predictions
