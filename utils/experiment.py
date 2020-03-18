from abc import abstractmethod
from treeRNN.trainer import train_and_validate, test
from utils.utils import set_initial_seed, get_logger, create_datatime_dir
import os
import torch as th
from preprocessing.dataset import ListDataset
from treeRNN.cells import TypedTreeCell
import pickle


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

    def run_training(self):
        training_config = self.config.training_config

        # initialise random seed
        if 'seed' in self.config.others_config:
            set_initial_seed(self.config.others_config.seed)

        # set the device
        device = self.__get_device__()

        trainset, valset = self.__load_training_data__()

        m = self.__create_model__()

        opt = self.__get_optimiser__(m)

        # train and validate
        best_dev_metric, best_model, info_training = train_and_validate(m, self.__get_loss_function__(), opt, trainset, valset,
                                                                        batcher_fun=self.__get_batcher_function__(),
                                                                        logger=self.logger.getChild('train'),
                                                                        **training_config)

        best_model_weights = best_model.state_dict()
        th.save(best_model_weights, os.path.join(self.output_dir, 'model_weight.pth'))
        th.save(info_training, os.path.join(self.output_dir, 'info_training.pth'))

        return best_dev_metric, best_model.state_dict()

    def run_test(self, state_dict):
        training_config = self.config.training_config

        testset = self.__load_test_data__()

        m = self.__create_model__()
        m.load_state_dict(state_dict)

        device = self.__get_device__()

        test_metrics, test_prediction = test(m, testset,
                                             batcher_fun=self.__get_batcher_function__(),
                                             logger=self.logger.getChild('test'),
                                             metric_class_list=training_config.metrics_class,
                                             batch_size=training_config.batch_size)

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
            exp = self.experiment_class('run_{}'.format(id), c, sub_out_dir, logger=None)
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
            fw.write(best_config.to_json())

        self.logger.info('Testing the best configuration.')
        test_exp = self.experiment_class('test_best_model', best_config, self.output_dir, self.logger.getChild('best_model_testing'))
        test_metrics, test_prediction = test_exp.run_test(best_ms_model_weight)

        th.save(test_prediction, os.path.join(self.output_dir, 'best_model_prediction.pth'))

    @staticmethod
    def __create_output_dir__(par_dir, id):
        p = os.path.join(par_dir, 'run_{}'.format(id))
        os.makedirs(p)
        return p
