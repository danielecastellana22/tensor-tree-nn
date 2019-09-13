from tqdm import tqdm
import torch as th
from experiments.execution_utils import get_sub_logger
import copy
from .metrics import ValueMetric, TreeMetric
import time


def __evaluate_model__(model, extract_batch_data, dataloader, metrics_class, pbar, batch_size):

    predictions = []
    eval_time = 0
    metrics = []
    for c in metrics_class:
        metrics.append(c())

    model.eval()
    for step, batch in enumerate(dataloader):

        t = time.time()
        in_data, out_data, graph = extract_batch_data(batch)
        with th.no_grad():
            out = model(*in_data)
        predictions.append(out)

        # update all metrics
        for v in metrics:
            if isinstance(v, ValueMetric):
                v.update_metric(out, out_data)

            if isinstance(v, TreeMetric):
                v.update_metric(out, out_data, graph)
        eval_time += (time.time() - t)

        pbar.update(min(batch_size, pbar.total - pbar.n))

    pbar.close()

    return metrics, eval_time, predictions


def __train_model__(model, trainset):
    raise Exception('This function is not implemented yet!')


def train_and_validate(model, extract_batch_data, loss_function, optimizer, trainset, devset, device, metrics_class, batch_size=25, n_epochs=200, early_stopping_patience=20, evaluate_on_training_set=False):
    logger = get_sub_logger('train_and_validate')

    best_dev_metric = None
    trainloader = trainset.get_loader(batch_size, device, shuffle=True)
    devloader = devset.get_loader(batch_size, device)

    best_metrics = []
    best_epoch = -1
    best_model = None

    dev_metrics = {}
    tr_metrics = {}
    for c in metrics_class:
        dev_metrics[c.__name__] = []
        tr_metrics[c.__name__] = []

    tr_forw_time_list = []
    tr_backw_time_list = []
    dev_val_time_list = []

    for epoch in range(n_epochs):
        model.train()

        tr_forw_time = 0
        tr_backw_time = 0

        with tqdm(total=len(trainset), desc='Training epoch ' + str(epoch) + ': ') as pbar:
            for step, batch in enumerate(trainloader):

                t = time.time()
                in_data, out_data,  _ = extract_batch_data(batch)
                model_output = model(*in_data)
                loss = loss_function(model_output, out_data)
                tr_forw_time += (time.time() - t)

                t = time.time()
                optimizer.zero_grad()
                loss.backward()
                #oo = optimizer.param_groups[0]['params'][0]
                #logger.debug('Gradient stats: sum: {:.10f}\tmin: {:.10f}\tmax: {:.10f}\tnorm: {:.10f}'.format(oo.sum(), oooptimizer.param_groups[0]['params'][3].grad.min(), oo.max(), oo.norm()))
                #ppp = optimizer.param_groups[2]['params'][1].data.clone()
                optimizer.step()
                #up = optimizer.param_groups[2]['params'][1].data - ppp

                tr_backw_time += (time.time() - t)

                pbar.update(min(batch_size, pbar.total-pbar.n))

        if evaluate_on_training_set:
            # eval on tr set
            pbar = tqdm(total=len(trainset), desc='Evaluate epoch ' + str(epoch) + ' on training set: ')
            metrics, _, _ = __evaluate_model__(model, extract_batch_data, trainloader, metrics_class, pbar, batch_size)

            # print tr metrics
            s = "Evaluation on training set: Epoch {:03d} | ".format(epoch)
            for v in metrics:
                v.finalise_metric()
                s += str(v) + " | "
                tr_metrics[type(v).__name__].append(v.get_value())
            logger.info(s)

        # eval on dev set
        pbar = tqdm(total=len(devset), desc='Evaluate epoch ' + str(epoch) + ' on dev set: ')
        metrics, eval_dev_time, _ = __evaluate_model__(model, extract_batch_data, devloader, metrics_class, pbar, batch_size)

        # print dev metrics
        s = "Evaluation on dev set: Epoch {:03d} | ".format(epoch)
        for v in metrics:
            v.finalise_metric()
            s += str(v) + " | "
            dev_metrics[type(v).__name__].append(v.get_value())
        logger.info(s)

        # early stopping
        if epoch == 0:
            best_dev_metric = metrics[0].get_value()
            best_epoch = epoch
            best_metrics = metrics
            best_model = copy.deepcopy(model)
        else:
            # the metrics in poisiton 0 is the one used to validate the model
            if metrics[0].is_better_than(best_dev_metric):
                best_dev_metric = metrics[0].get_value()
                best_epoch = epoch
                best_metrics = metrics
                best_model = copy.deepcopy(model)
                logger.info('Epoch {:03d}: New optimum found'.format(epoch))
            else:
                # early stopping
                if best_epoch <= epoch - early_stopping_patience:
                    break

        # lr decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(1e-5, param_group['lr'] * 0.99)  # 10

        tr_forw_time_list.append(tr_forw_time)
        tr_backw_time_list.append(tr_backw_time)
        dev_val_time_list.append(eval_dev_time)

    # build vocabulary for the result
    info_training = {}
    info_training['best_epoch'] = best_epoch
    info_training['tr_metrics'] = tr_metrics
    info_training['dev_metrics'] = dev_metrics
    info_training['tr_forward_time'] = tr_forw_time_list[:best_epoch+1]
    info_training['tr_bakcward_time'] = tr_backw_time_list[:best_epoch+1]
    info_training['dev_eval_time'] = dev_val_time_list[:best_epoch+1]

    return best_model, info_training


def test(model, extract_batch_data, testset, device, metrics_class, batch_size=25):
    logger = get_sub_logger('test')

    testloader = testset.get_loader(batch_size, device)

    pbar = tqdm(total=len(testset), desc='Evaluate on test set: ')
    metrics, eval_dev_time, predictions = __evaluate_model__(model, extract_batch_data, testloader, metrics_class, pbar, batch_size)

    test_metrics = {}
    # print metrics
    s = "Test: "
    for v in metrics:
        v.finalise_metric()
        s += str(v) + " | "
        test_metrics[type(v).__name__] = v.get_value()

    logger.info(s)

    return test_metrics, predictions
