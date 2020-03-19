from abc import abstractmethod
from treeRNN.trainer import train_and_validate, test
from utils.utils import set_initial_seed, get_logger, create_datatime_dir, to_json_file
import os
import torch as th
from preprocessing.dataset import ListDataset
from treeRNN.cells import TypedTreeCell
import pickle
import numpy as np


# base class for all experiments
class Experiment:

    def __init__(self, config, output_dir, logger):

        self.config = config
        self.output_dir = output_dir
        self.logger = logger

        # save config
        with open(os.path.join(output_dir, 'run_config.json'), 'w') as fw:
            fw.write(config.to_json())

    # this methods return trainset, valset
    def __load_training_data__(self):
        dataset_config = self.config.dataset_config

        data_dir = dataset_config.data_dir

        with open(os.path.join(data_dir, 'train.pkl'), 'rb') as rf:
            trainset = ListDataset(pickle.load(rf))

        with open(os.path.join(data_dir, 'validation.pkl'), 'rb') as rf:
            valset = ListDataset(pickle.load(rf))

        return trainset, valset

    def __load_test_data__(self):
        dataset_config = self.config.dataset_config
        data_dir = dataset_config.data_dir

        with open(os.path.join(data_dir, 'test.pkl'), 'rb') as rf:
            testset = ListDataset(pickle.load(rf))

        return testset

    def __load_input_embeddings__(self):
        if 'input_model_config' in self.config:
            input_model_config = self.config.input_model_config
            if 'pretrained_embs' in input_model_config:
                with open(input_model_config.pretrained_embs, 'rb') as rf:
                    np_array = pickle.load(rf)
                return th.tensor(np_array, dtype=th.float32)

        return None

    def __load_type_embeddings__(self):
        if 'type_model_config' in self.config:
            type_model_config = self.config.type_model_config
            if 'pretrained_embs' in type_model_config:
                with open(type_model_config.pretrained_embs, 'rb') as rf:
                    np_array = pickle.load(rf)
                return th.tensor(np_array, dtype=th.float32)
        return None

    @abstractmethod
    def __create_model__(self):
        pass

    @abstractmethod
    def __get_optimiser__(self, model):
        pass

    @abstractmethod
    def __get_loss_function__(self):
        pass

    @abstractmethod
    def __get_batcher_function__(self):
        pass

    def __create_cell_module__(self):
        tree_model_config = self.config.tree_model_config
        cell_config = tree_model_config.cell_config

        cell_class = cell_config.cell_class
        is_typed = cell_config.typed if hasattr(cell_config, 'typed') else False
        cell_params = cell_config.cell_params

        if not is_typed:
            cell = cell_class(x_size=tree_model_config.x_size,
                              h_size=tree_model_config.h_size,
                              **cell_params)
        else:
            # the key can be used to assign particular
            n_types = cell_config.num_types
            cell = TypedTreeCell(x_size=tree_model_config.x_size,
                                 h_size=tree_model_config.h_size,
                                 cell_class=cell_class,
                                 cells_params_list=[cell_params for i in range(n_types)],
                                 share_input_matrices=cell_config.share_input_matrices)

        return cell

    def __get_device__(self):
        dev = self.config.others_config.gpu
        cuda = dev >= 0
        device = th.device('cuda:{}'.format(dev)) if cuda else th.device('cpu')
        if cuda:
            th.cuda.set_device(dev)
        else:
            th.set_num_threads(-dev)
        return device

    def run_training(self, metric_class_list):
        training_config = self.config.training_config

        # initialise random seed
        if 'seed' in self.config.others_config:
            set_initial_seed(self.config.others_config.seed)

        trainset, valset = self.__load_training_data__()

        m = self.__create_model__()

        opt = self.__get_optimiser__(m)

        # train and validate
        best_val_metrics, best_model, info_training = train_and_validate(m, self.__get_loss_function__(), opt, trainset, valset,
                                                                         batcher_fun=self.__get_batcher_function__(),
                                                                         logger=self.logger.getChild('train'),
                                                                         metric_class_list=metric_class_list,
                                                                         **training_config)

        best_model_weights = best_model.state_dict()
        th.save(best_model_weights, os.path.join(self.output_dir, 'model_weight.pth'))
        to_json_file(info_training, os.path.join(self.output_dir, 'info_training.json'))

        return best_val_metrics

    def run_test(self, state_dict, metric_class_list):
        training_config = self.config.training_config

        testset = self.__load_test_data__()

        m = self.__create_model__()
        m.load_state_dict(state_dict)

        test_metrics, test_prediction = test(m, testset,
                                             batcher_fun=self.__get_batcher_function__(),
                                             logger=self.logger.getChild('test'),
                                             metric_class_list=metric_class_list,
                                             batch_size=training_config.batch_size)

        return test_metrics, test_prediction


class ExperimentRunner:

    # TODO: add recovery strategy
    # TODO: param to enable multiprocessing
    def __init__(self, experiment_class, output_dir, num_run, metric_class_list, config_list):
        self.experiment_class = experiment_class
        self.config_list = config_list
        self.num_run = num_run
        self.output_dir = create_datatime_dir(output_dir)
        self.logger = get_logger('runner', self.output_dir, write_on_console=True)
        self.metric_class_list = metric_class_list

    def run(self):

        self.logger.info('Model selection starts: {} configurations to run {} times.'.format(len(self.config_list), self.num_run))
        all_best_dev_metrics = []
        for i_config, c in enumerate(self.config_list):
            all_best_dev_metrics.append([])
            for i_run in range(self.num_run):
                self.logger.info('Configuration {} Run {}.'.format(i_config, i_run))

                exp_id = 'c{}_r{}'.format(i_config, i_run)
                exp_out_dir = self.__create_output_dir__(i_config, i_run)

                exp_logger = get_logger(exp_id, exp_out_dir, write_on_console=False)
                exp = self.experiment_class(c, exp_out_dir, exp_logger)
                val_metrics = exp.run_training(self.metric_class_list)

                val_metrics_dict = {type(x).__name__: x.get_value() for x in val_metrics}
                all_best_dev_metrics[i_config].append(val_metrics_dict)

                self.logger.info('Configuration {} Run {}: {}.'.format(i_config, i_run, ' | '.join(map(str, val_metrics))))

        self.logger.info('Model selection finished.')

        ms_dev_results = np.array([[y[self.metric_class_list[0].__name__] for y in x] for x in all_best_dev_metrics])
        to_json_file(all_best_dev_metrics, os.path.join(self.output_dir, 'all_validation_results.json'))

        ms_dev_mean = np.mean(ms_dev_results, axis=1)

        if self.metric_class_list[0].HIGHER_BETTER:
            best_config_id = np.argmax(ms_dev_mean)
        else:
            best_config_id = np.argmin(ms_dev_mean)

        self.logger.info('Configuration {} is the best one! Validation Score: {}'.format(best_config_id, ms_dev_mean[best_config_id]))

        # save best config
        self.logger.info('Saving best configuration.')
        with open(os.path.join(self.output_dir, 'best_config.json'), 'w') as fw:
            fw.write(self.config_list[best_config_id].to_json())

        # load best weight from the first run
        self.logger.info('Load best model weight.')
        best_model_weight = self.__get_model_weight__(best_config_id, 0)

        self.logger.info('Testing the best configuration.')
        test_exp = self.experiment_class(self.config_list[best_config_id], self.output_dir,
                                         self.logger.getChild('best_model_testing'))
        test_metrics, test_prediction = test_exp.run_test(best_model_weight, self.metric_class_list)
        self.logger.info('Test Results: {}.'.format(' | '.join(map(str, test_metrics))))

        th.save(test_prediction, os.path.join(self.output_dir, 'best_model_prediction.pth'))

    def __get_model_weight__(self, id_config, n_run):
        sub_dir = os.path.join(self.output_dir, 'conf_{}/run_{}'.format(id_config, n_run))
        return th.load(os.path.join(sub_dir, 'model_weight.pth'))

    def __create_output_dir__(self, id_config, n_run):
        p = os.path.join(self.output_dir, 'conf_{}/run_{}'.format(id_config, n_run))
        os.makedirs(p)
        return p


