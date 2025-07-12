from tqdm import tqdm
import torch as th
import copy
import time
from exputils.configurations import create_object_from_config


def __update_metrics__(model_pred, in_data, out_data, metrics):
    # update all metrics
    for v in metrics:
        v.update_metric(model_pred, out_data, *in_data)


class BaseTrainer:

    # TODO: transform trainer in an event dispatcher
    def __init__(self, debug_mode, logger, n_epochs,
                 early_stopping_patience=-1, evaluate_on_training_set=True, eps_loss=None,
                 min_n_epochs=1):
        self.debug_mode = debug_mode
        self.logger = logger
        self.n_epochs = n_epochs
        # TODO: check when these values are not set
        self.early_stopping_patience = early_stopping_patience
        self.evaluate_on_training_set = evaluate_on_training_set
        self.eps_loss = eps_loss
        self.min_n_epochs = min_n_epochs

    def __on_training_start__(self, logger, model, train_loader, val_loader, metric_class_list):
        pass

    def __training_step__(self, model, in_data, out_data):
        raise NotImplementedError('This method must be specified in the subclass')

    def __on_epoch_ends__(self, model):
        pass

    def __early_stopping_on_loss__(self, tot_loss):
        return False

    def train_and_validate(self, model, train_loader, val_loader, metric_class_list):

        logger = self.logger.getChild('train')

        self.__on_training_start__(logger, model, train_loader, val_loader,metric_class_list)

        best_val_metrics = None
        best_epoch = -1
        best_model = None

        val_metrics_dict = {}
        tr_metrics_dict = {}
        for c in metric_class_list:
            val_metrics_dict[c.get_name()] = []
            tr_metrics_dict[c.get_name()] = []

        tr_forward_time_list = []
        tr_backward_time_list = []
        val_time_list = []

        for epoch in range(1, self.n_epochs+1):
            model.train()

            tr_forward_time = 0
            tr_backward_time = 0

            # TODO: implement print loss every tot. Can be useful for big dataset
            logger.debug('START TRAINING EPOCH {}.'.format(epoch))

            loss_to_print = 0
            tot_loss = 0

            # initialise online training metrics. ONLY NEURAL MODEL CAN COMPUTE THEM
            compute_tr_metrics_online = False
            online_tr_metrics = []
            for c in metric_class_list:
                online_tr_metrics.append(c())

            for batch in tqdm(train_loader, desc='Training epoch ' + str(epoch) + ': ', disable=not self.debug_mode):
                in_data = batch[0]
                out_data = batch[1]
                if self.debug_mode:
                    with th.autograd.detect_anomaly():
                        dict_out = self.__training_step__(model=model, in_data=in_data, out_data=out_data)
                else:
                    dict_out = self.__training_step__(model=model, in_data=in_data, out_data=out_data)

                tot_loss += dict_out['loss']
                loss_to_print += dict_out['loss']
                tr_forward_time += dict_out['tr_forward_time']
                tr_backward_time += dict_out['tr_backward_time']

                if dict_out['predictions'] is not None:
                    compute_tr_metrics_online = True
                    # evaluate metrics
                    __update_metrics__(dict_out['predictions'], in_data, out_data, online_tr_metrics)

            # finalise tr_metrics
            s = "End training: Epoch {:3d} | Tot. Loss: {:4.3f}".format(epoch, tot_loss)
            if compute_tr_metrics_online:
                s += ' | '
                for v in online_tr_metrics:
                    v.finalise_metric()
                    s += str(v) + " | "
                    tr_metrics_dict[v.get_name()].append(v.get_value())
            self.logger.info(s)
            self.__on_epoch_ends__(model)

            # if tr_metrics are not computed onlin, we visit again the whole training set.
            # PROBABILISTIC MODEL NEEDS THIS
            if not compute_tr_metrics_online and self.evaluate_on_training_set:
                logger.debug("START EVALUATION ON TRAINING SET")
                # eval on tr set
                metrics, _, _ = self.__evaluate_model__(model, train_loader, metric_class_list,
                                                        'Evaluate epoch ' + str(epoch) + ' on training set: ')

                # print tr metrics
                s = "Evaluation on training set: Epoch {:03d} | ".format(epoch)
                for v in metrics:
                    s += str(v) + " | "
                    tr_metrics_dict[v.get_name()].append(v.get_value())
                logger.info(s)

            # eval on validation set
            eval_val_time = 0
            if epoch > self.min_n_epochs:
                logger.debug("START EVALUATION ON VALIDATION SET")
                metrics, eval_val_time, _ = self.__evaluate_model__(model, val_loader, metric_class_list,
                                                                    'Evaluate epoch ' + str(epoch) + ' on validation set: ')

                # print validation metrics
                s = "Evaluation on validation set: Epoch {:03d} | ".format(epoch)
                for v in metrics:
                    s += str(v) + " | "
                    val_metrics_dict[v.get_name()].append(v.get_value())
                logger.info(s)

                # select best model
                if best_val_metrics is None:
                    best_val_metrics = copy.deepcopy(metrics)
                    best_epoch = epoch
                    best_model = copy.deepcopy(model)
                else:
                    # the metrics in position 0 is the one used to validate the model
                    if metrics[0].is_better_than(best_val_metrics[0]):
                        best_val_metrics = copy.deepcopy(metrics)
                        best_epoch = epoch
                        best_model = copy.deepcopy(model)
                        logger.info('Epoch {:03d}: New optimum found'.format(epoch))
                    else:
                        # early stopping
                        if best_epoch <= epoch - self.early_stopping_patience or \
                           self.__early_stopping_on_loss__(tot_loss):
                            break

            tr_forward_time_list.append(tr_forward_time)
            tr_backward_time_list.append(tr_backward_time)
            val_time_list.append(eval_val_time)

        # print best results
        s = "Best found in Epoch {:03d} | ".format(best_epoch)
        for v in best_val_metrics:
            s += str(v) + " | "
        logger.info(s)

        # build vocabulary for the result
        info_training = {
            'best_epoch': best_epoch,
            'tr_metrics': tr_metrics_dict,
            'val_metrics': val_metrics_dict,
            'tr_forward_time': tr_forward_time_list,
            'tr_backward_time': tr_backward_time_list,
            'val_eval_time': val_time_list}

        return best_val_metrics, best_model, info_training

    def test(self, model, test_loader, metric_class_list):

        logger = self.logger.getChild('test')
        metrics, _, predictions = self.__evaluate_model__(model, test_loader, metric_class_list,
                                                          'Evaluate on test set: ')

        # print metrics
        s = "Test: "
        for v in metrics:
            s += str(v) + " | "

        logger.info(s)

        return metrics, predictions

    def __evaluate_model__(self, model, data_loader, metric_class_list, desc):
        eval_time = 0
        metrics = []
        for c in metric_class_list:
            metrics.append(c())

        predictions = []
        model.eval()
        for batch in tqdm(data_loader, desc=desc, disable=not self.debug_mode):
            t = time.time()
            in_data = batch[0]
            out_data = batch[1]
            with th.no_grad():
                out = model(*in_data)

            predictions.append(out)

            __update_metrics__(out, in_data, out_data, metrics)

            eval_time += (time.time() - t)

        for v in metrics:
            v.finalise_metric()

        return metrics, eval_time, th.stack(predictions, dim=0)


class NeuralTrainer(BaseTrainer):

    def __init__(self, optimiser, loss_function, **kwargs):
        super().__init__(**kwargs)
        self.optimiser_config = optimiser
        self.optimiser = None
        self.loss_function = create_object_from_config(loss_function)

    def __on_training_start__(self, logger, model, train_loader, val_loader, metric_class_list):
        self.optimiser = create_object_from_config(self.optimiser_config, params=[x for x in model.parameters() if x.requires_grad])

    def __training_step__(self, model, in_data, out_data):
        t = time.time()

        model_output = model(*in_data)

        loss = self.loss_function(model_output, out_data)
        tr_forward_time = (time.time() - t)

        t = time.time()
        self.optimiser.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10)
        self.optimiser.step()

        tr_backward_time = (time.time() - t)

        return {'loss': loss.item(), 'predictions': model_output,
                'tr_forward_time': tr_forward_time, 'tr_backward_time': tr_backward_time}


class EMTrainer(BaseTrainer):

    def __training_step__(self, model, in_data, out_data):
        t = time.time()
        log_like = model(*in_data, out_data=out_data)
        tr_forward_time = (time.time() - t)

        return {'loss': log_like.item(), 'predictions': {}, 'tr_forward_time': tr_forward_time, 'tr_backward_time': 0}

    def __on_epoch_ends__(self, model):
        model.m_step()

    def __early_stopping_on_loss__(self, tot_loss):
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

                # stop after 5 epoch with decreasing log-likelihood
                return self.n_epoch_decr >= 5
            else:
                self.n_epoch_decr = 0
                out = (tot_loss - self.prev_loss) < self.eps_loss
                self.prev_loss = tot_loss
                return out
