from tqdm import tqdm
import torch as th
import copy
from experiments.metrics import ValueMetricUpdate, TreeMetricUpdate
import time
from torch.utils.data import DataLoader


class BaseTrainer:

    def __init__(self, debug_mode, logger):
        self.debug_mode = debug_mode
        self.logger = logger

    def __training_step__(self, **kwargs):
        raise NotImplementedError('This methos must be specified in the subclass')

    def __on_epoch_ends__(self, model, **kwargs):
        pass

    def __early_stopping_on_loss__(self, tot_loss, eps_loss):
        return False

    def train_and_validate(self, **kwargs):

        model = kwargs.pop('model')
        #loss_function, optimizer,
        trainset = kwargs.pop('trainset')
        valset = kwargs.pop('valset')
        collate_fun = kwargs.pop('collate_fun')
        metric_class_list = kwargs.pop('metric_class_list')
        logger = self.logger.getChild('train')
        batch_size = kwargs.pop('batch_size')
        n_epochs = kwargs.pop('n_epochs')
        early_stopping_patience = kwargs.pop('early_stopping_patience')
        evaluate_on_training_set = kwargs.pop('evaluate_on_training_set') if 'evaluate_on_training_set' in kwargs else False
        eps_loss = kwargs.pop('eps_loss', None)

        train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fun, shuffle=True, num_workers=0)
        val_loader = DataLoader(valset, batch_size=batch_size, collate_fn=collate_fun, shuffle=False, num_workers=0)

        best_val_metrics = None
        best_epoch = -1
        best_model = None

        val_metrics = {}
        tr_metrics = {}
        for c in metric_class_list:
            val_metrics[c.get_name()] = []
            tr_metrics[c.get_name()] = []

        tr_forw_time_list = []
        tr_backw_time_list = []
        val_time_list = []

        for epoch in range(1, n_epochs+1):
            model.train()

            tr_forw_time = 0
            tr_backw_time = 0

            # TODO: implement print loss every tot. Can be useful for big dataset
            logger.debug('START TRAINING EPOCH {}.'.format(epoch))
            with tqdm(total=len(trainset), desc='Training epoch ' + str(epoch) + ': ', disable=not self.debug_mode) as pbar:

                print_every = pbar.total // 100
                loss_to_print = 0
                tot_loss = 0
                n=0
                for step, batch in enumerate(train_loader):

                    in_data = batch[0]
                    out_data = batch[1]
                    loss, f_time, b_time = self.__training_step__(model=model, in_data=in_data, out_data=out_data, **kwargs)
                    tot_loss += loss
                    loss_to_print += loss
                    tr_forw_time += f_time
                    tr_backw_time += b_time

                    n += min(batch_size, pbar.total - n)
                    pbar.update(min(batch_size, pbar.total - n))

            self.logger.info("End training: Epoch {:3d} | Tot. Loss: {:4.3f}".format(epoch, tot_loss))
            self.__on_epoch_ends__(model)

            if evaluate_on_training_set:
                logger.debug("START EVALUATION ON TRAINING SET")
                # eval on tr set
                pbar = tqdm(total=len(trainset), desc='Evaluate epoch ' + str(epoch) + ' on training set: ',
                            disable=not self.debug_mode)
                metrics, _, _ = self.__evaluate_model__(model, train_loader, metric_class_list, pbar, batch_size)

                # print tr metrics
                s = "Evaluation on training set: Epoch {:03d} | ".format(epoch)
                for v in metrics:
                    s += str(v) + " | "
                    tr_metrics[v.get_name()].append(v.get_value())
                logger.info(s)

            # eval on validation set
            logger.debug("START EVALUATION ON VALIDATION SET")
            pbar = tqdm(total=len(valset), desc='Evaluate epoch ' + str(epoch) + ' on validation set: ',
                        disable=not self.debug_mode)
            metrics, eval_val_time, _ = self.__evaluate_model__(model, val_loader, metric_class_list, pbar, batch_size)

            # print validation metrics
            s = "Evaluation on validation set: Epoch {:03d} | ".format(epoch)
            for v in metrics:
                s += str(v) + " | "
                val_metrics[v.get_name()].append(v.get_value())
            logger.info(s)

            # select best model
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
                    if best_epoch <= epoch - early_stopping_patience or \
                       self.__early_stopping_on_loss__(tot_loss, eps_loss):
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

    def test(self, model, testset, collate_fun, metric_class_list, batch_size):

        logger = self.logger.getChild('test')
        testloader = DataLoader(testset, batch_size=batch_size, collate_fn=collate_fun, shuffle=False, num_workers=0)

        pbar = tqdm(total=len(testset), desc='Evaluate on test set: ', disable=not self.debug_mode)
        metrics, _, predictions = self.__evaluate_model__(model, testloader, metric_class_list, pbar, batch_size)

        # print metrics
        s = "Test: "
        for v in metrics:
            s += str(v) + " | "

        logger.info(s)

        return metrics, predictions

    def __evaluate_model__(self, model, dataloader, metric_class_list, pbar, batch_size):
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


class NeuralTrainer(BaseTrainer):

    def __training_step__(self, model, in_data, out_data, loss_function, optimiser):
        t = time.time()

        model_output = model(*in_data)
        loss = loss_function(model_output, out_data)
        tr_forw_time = (time.time() - t)

        t = time.time()
        optimiser.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10)
        optimiser.step()

        tr_backw_time = (time.time() - t)

        return loss.item(), tr_forw_time, tr_backw_time


class EMTrainer(BaseTrainer):

    def __training_step__(self, model, in_data, out_data):
        t = time.time()
        loglike = model(*in_data, out_data=out_data)
        tr_forw_time = (time.time() - t)

        # t = time.time()
        # model.accumulate_posterior(in_data, out_data)
        # tr_backw_time = (time.time() - t)

        return loglike.item(), tr_forw_time, 0

    def __on_epoch_ends__(self, model, **kwargs):
        model.m_step()

    def __early_stopping_on_loss__(self, tot_loss, eps_loss):
        if not hasattr(self, 'prev_loss'):
            self.prev_loss = tot_loss
            return False
        else:
            if (tot_loss - self.prev_loss) < 0:
                self.logger.getChild('train').warning('Negative Log-Likelihood is decreasing!')
                if hasattr(self, 'n_epoch_decr'):
                    self.n_epoch_decr += 1
                else:
                    self.n_epoch_decr = 1

                # stop after 5 epoch with decreasing loglikelihood
                return self.n_epoch_decr >= 5
            else:
                self.n_epoch_decr = 0
                out = (tot_loss - self.prev_loss) < eps_loss
                self.prev_loss = tot_loss
                return out




