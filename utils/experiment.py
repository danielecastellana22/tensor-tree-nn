from abc import abstractmethod
from treeRNN.trainer import train_and_validate, test
from utils.utils import set_initial_seed, get_logger, create_datatime_dir
import os
import torch as th
from treeRNN.cells import TypedTreeCell


# base class for all experiments
class Experiment:

    def __init__(self, id, config, output_dir, logger):

        #if output_dir is None and logger is None:
        #    raise ValueError('At least one between output_dir and logger must be specified.')

        self.id = id
        self.config = config
        self.output_dir = output_dir
        if logger is None:
            self.logger = get_logger(id, output_dir, write_on_console=False)
        else:
            self.logger = logger

    @abstractmethod
    # TODO: dataset should be preprocessed and here eveything should be setted
    def __load_dataset__(self, load_embs):
        pass

    @abstractmethod
    def __create_model__(self, trainset, in_pretrained_embs, type_pretrained_embs):
        pass

    @abstractmethod
    def __get_optimiser__(self, model):
        pass

    @abstractmethod
    def __get_loss_function__(self):
        pass

    def __create_cell_module__(self, max_out_degree, n_type):
        tree_model_config = self.config.tree_model_config
        cell_config = tree_model_config['cell_config']

        cell_class = cell_config['cell_class']
        is_typed = cell_config['typed'] if 'typed' in cell_config else False
        cell_params = cell_config['cell_params']

        if not is_typed:
            cell = cell_class(x_size=tree_model_config['x_size'],
                              h_size=tree_model_config['h_size'],
                              max_output_degree=max_out_degree,
                              **cell_params)
        else:
            # the key can be used to assign particular
            cell = TypedTreeCell(x_size=tree_model_config['x_size'],
                                 h_size=tree_model_config['h_size'],
                                 cell_class=cell_class,
                                 cells_params_list=[cell_params for i in range(n_type)],
                                 share_input_matrices = cell_config['share_input_matrices'])

        return cell

    def __get_device__(self):
        dev = self.config.training_config['gpu']
        cuda = dev >= 0
        device = th.device('cuda:{}'.format(dev)) if cuda else th.device('cpu')
        if cuda:
            th.cuda.set_device(dev)
        else:
            th.set_num_threads(-dev)
        return device

    def run_training(self):
        training_config = self.config.training_config

        # initialise random seed
        if 'seed' in training_config:
            set_initial_seed(training_config['seed'])

        # set the device
        device = self.__get_device__()

        trainset, devset, testset, in_pretrained_embs, type_pretrained_embs = self.__load_dataset__(load_embs=True)

        m = self.__create_model__(trainset, in_pretrained_embs, type_pretrained_embs)

        opt = self.__get_optimiser__(m)

        # train and validate
        best_dev_metric, best_model, info_training = train_and_validate(m, self.__get_loss_function__(), opt, trainset, devset, device,
                                                       logger=self.logger.getChild('train'),
                                                       metric_class_list=training_config['metrics_class'],
                                                       batch_size=training_config['batch_size'],
                                                       n_epochs=training_config['n_epochs'],
                                                       early_stopping_patience=training_config['early_stopping_patience'],
                                                       evaluate_on_training_set=training_config['evaluate_on_training_set'] if 'evaluate_on_training_set' in training_config else True)

        best_model_weights = best_model.state_dict()
        th.save(best_model_weights, os.path.join(self.output_dir, 'model_weight.pth'))
        th.save(info_training, os.path.join(self.output_dir, 'info_training.pth'))

        return best_dev_metric, best_model.state_dict()

    def run_test(self, state_dict):
        training_config = self.config.training_config

        trainset, devset, testset, in_pretrained_embs, type_pretrained_embs = self.__load_dataset__(load_embs=True)

        m = self.__create_model__(trainset, in_pretrained_embs, type_pretrained_embs)
        m.load_state_dict(state_dict)

        device = self.__get_device__()

        test_metrics, test_prediction = test(m, testset, device, logger=self.logger.getChild('test'),
                                        metric_class_list=training_config['metrics_class'],
                                        batch_size=training_config['batch_size'])

        return test_metrics, test_prediction


class ExperimentRunner:

    # TODO: add recovery strategy
    # TODO: maybe other params to enable multiprocessing
    def __init__(self, experiment_class, output_dir, config_list):
        self.experiment_class = experiment_class
        self.config_list = config_list
        self.output_dir = create_datatime_dir(output_dir)
        self.logger = get_logger('runner', self.output_dir, write_on_console=True)

    def run(self):
        best_ms_metric = None
        best_ms_model_weight = None
        best_config = None

        self.logger.info('Model selection starts: {} configuration to run.'.format(len(self.config_list)))
        for id, c in enumerate(self.config_list):
            self.logger.info('Running configuration {}.'.format(id))
            sub_out_dir = self.__create_output_dir__(self.output_dir, id)
            exp = self.experiment_class('run_{}'.format(id), c, sub_out_dir)
            val_metric, model_weight = exp.run_training()
            self.logger.info('Configuration {} score: {}.'.format(id, str(val_metric)))
            if best_ms_metric is None or val_metric.is_better_than(best_ms_metric):
                self.logger.info('Configuration {} is the new optimum!'.format(id))
                best_ms_metric = val_metric
                best_ms_model_weight = model_weight
                best_config = c

        self.logger.info('Model selection finished.')

        self.logger.info('Saving best model weight.')
        th.save(best_ms_model_weight, os.path.join(self.output_dir, 'best_model_weight.pth'))
        with open(os.path.join(self.output_dir, 'best_config.json'), 'w') as fw:
            fw.write(str(best_config))

        self.logger.info('Testing the best configuration.')
        test_exp = self.experiment_class('test_best_model', best_config, self.output_dir, self.logger.getChild('best_model_testing'))
        test_metrics, test_prediction = test_exp.run_test(best_ms_model_weight)

        th.save(test_prediction, os.path.join(self.output_dir, 'best_model_prediction.pth'))

    @staticmethod
    def __create_output_dir__(par_dir, id):
        p = os.path.join(par_dir, 'run_{}'.format(id))
        os.makedirs(p)
        return p
