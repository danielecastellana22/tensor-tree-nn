import os
from utils.misc import set_initial_seed, string2class
from utils.serialization import to_json_file, from_pkl_file, to_torch_file
from experiments.config import create_object_from_config
import torch as th
from preprocessing.dataset import ListDataset
from abc import abstractmethod


class CollateFun:

    def __init__(self, device, **kwargs):
        self.device = device

    @abstractmethod
    def __call__(self, tuple_data):
        raise NotImplementedError('Must be implemented in subclasses')


# class for all experiments
class Experiment:

    def __init__(self, config, output_dir, logger, debug_mode):

        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        self.debug_mode = debug_mode

        # save config
        to_json_file(self.config, os.path.join(output_dir, 'config.json'))

    ####################################################################################################################
    # DATASET FUNCTIONS
    ####################################################################################################################
    # this methods return trainset, valset
    def __load_training_data__(self):
        dataset_config = self.config.dataset_config

        data_dir = dataset_config.data_dir

        # TODO: allows to load a dataset divided in multiple files (SNLI)
        trainset = ListDataset(from_pkl_file(os.path.join(data_dir, 'train.pkl')))
        valset = ListDataset(from_pkl_file(os.path.join(data_dir, 'validation.pkl')))

        return trainset, valset

    def __load_test_data__(self):
        dataset_config = self.config.dataset_config
        data_dir = dataset_config.data_dir

        testset = ListDataset(from_pkl_file(os.path.join(data_dir, 'test.pkl')))

        return testset

    ####################################################################################################################
    # MODULE FUNCTIONS
    ####################################################################################################################
    def __create_tree_module__(self):
        return create_object_from_config(self.config.tree_module_config)

    ####################################################################################################################
    # TRAINER FUNCTIONS
    ####################################################################################################################
    @staticmethod
    def __get_optimiser__(optim_config, model):
        optim_class = string2class(optim_config['class'])
        params_groups = dict(optim_config.params) if 'params' in optim_config else {}
        params_groups.update({'params': list(model.parameters())})

        return optim_class([params_groups])

    def __get_trainer__(self):
        return create_object_from_config(self.config.trainer_config, debug_mode=self.debug_mode, logger=self.logger)

    def __get_training_params__(self, model):
        d = {}
        d.update(self.config.trainer_config.training_params)

        if 'optimiser' in d:
            d['optimiser'] = self.__get_optimiser__(d['optimiser'], model)
        if 'loss_function' in d:
            d['loss_function'] = create_object_from_config(d['loss_function'])
        if 'batcher_fun' in d:
            d['batcher_fun'] = create_object_from_config(d['batcher_fun'], device=self.__get_device__())

        return d

    ####################################################################################################################
    # UTILS FUNCTIONS
    ####################################################################################################################
    def __get_device__(self):
        dev = self.config.others_config.gpu
        cuda = dev >= 0
        device = th.device('cuda:{}'.format(dev)) if cuda else th.device('cpu')
        if cuda:
            th.cuda.set_device(dev)

        return device

    def __save_test_model_params__(self, best_model):
        to_torch_file(best_model.state_dict(), os.path.join(self.output_dir, 'params_learned.pth'))

    ####################################################################################################################
    # TRAINING FUNCTION
    ####################################################################################################################
    def run_training(self, metric_class_list, do_test):
        # initialise random seed
        if 'seed' in self.config.others_config:
            seed = self.config.others_config.seed
        else:
            seed = -1
        seed = set_initial_seed(seed)
        self.logger.info('Seed set to {}.'.format(seed))

        trainset, valset = self.__load_training_data__()

        m = self.__create_tree_module__()
        # save number of parameters
        n_params_dict = {k: v.numel() for k, v in m.state_dict().items()}
        to_json_file(n_params_dict, os.path.join(self.output_dir, 'num_model_parameters.json'))

        trainer = self.__get_trainer__()
        training_params = self.__get_training_params__(m)

        # train and validate
        best_val_metrics, best_model, info_training = trainer.train_and_validate(model=m,
                                                                                 trainset=trainset,
                                                                                 valset=valset,
                                                                                 metric_class_list=metric_class_list,
                                                                                 **training_params)

        best_val_metrics_dict = {x.get_name(): x.get_value() for x in best_val_metrics}

        to_json_file(best_val_metrics_dict, os.path.join(self.output_dir, 'best_validation_metrics.json'))
        to_json_file(info_training, os.path.join(self.output_dir, 'info_training.json'))

        if not do_test:
            return best_val_metrics
        else:

            self.__save_test_model_params__(best_model)

            testset = self.__load_test_data__()

            test_metrics, test_prediction = trainer.test(best_model, testset,
                                                         collate_fun=training_params['batcher_fun'],
                                                         metric_class_list=metric_class_list,
                                                         batch_size=training_params['batch_size'])

            test_metrics_dict = {x.get_name(): x.get_value() for x in test_metrics}
            to_json_file(test_metrics_dict, os.path.join(self.output_dir, 'test_metrics.json'))
            to_torch_file(test_prediction, os.path.join(self.output_dir, 'test_prediction.pth'))
            return test_metrics