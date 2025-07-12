import os
import torch as th
from torch.utils.data import ConcatDataset
import copy

from exputils.utils import set_initial_seed
from exputils.serialisation import to_json_file, to_torch_file, from_torch_file
from exputils.configurations import create_object_from_config


# class for all experiments
class Experiment:

    def __init__(self, config, output_dir, logger, device, seed, debug_mode):

        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        self.debug_mode = debug_mode
        self.device = device
        # set device
        if self.device.startswith('cuda'):
            th.cuda.device(self.device)

        self.seed = set_initial_seed(seed)

        # save config
        to_json_file(self.config, os.path.join(output_dir, 'config.json'))

    ####################################################################################################################
    # DATASET FUNCTIONS
    ####################################################################################################################
    # this methods return trainset, valset

    def __get_dataset_loader__(self):
        return create_object_from_config(self.config.dataset_config, device=self.device)

    ####################################################################################################################
    # MODULE FUNCTIONS
    ####################################################################################################################
    def __create_exp_module__(self):
        m = create_object_from_config(self.config.exp_module_config)
        if 'params_path' in self.config.exp_module_config:
            run_dir = os.path.basename(os.path.normpath(self.output_dir))
            params_path = os.path.join(self.config.exp_module_config.params_path, run_dir)
            params_path = os.path.join(params_path, 'params_learned.pth')
            self.logger.warning('Loading model params from {}'.format(params_path))
            params = from_torch_file(params_path)
            m.load_state_dict(params)
        return m

    ####################################################################################################################
    # TRAINER FUNCTIONS
    ####################################################################################################################
    def __get_trainer__(self):
        return create_object_from_config(self.config.trainer_config, debug_mode=self.debug_mode, logger=self.logger)

    ####################################################################################################################
    # UTILS FUNCTIONS
    ####################################################################################################################
    def __save_test_model_params__(self, best_model):
        to_torch_file(best_model.state_dict(), os.path.join(self.output_dir, 'params_learned.pth'))

    ####################################################################################################################
    # TRAINING FUNCTION
    ####################################################################################################################
    def run_training(self, metric_class_list, do_test):
        # initialise random seed
        self.logger.info('Seed set to {}.'.format(self.seed))

        dataset_loader = self.__get_dataset_loader__()

        train_loader = dataset_loader.get_training_data_loader()
        val_loader = dataset_loader.get_validation_data_loader()

        m = self.__create_exp_module__()
        # save number of parameters
        n_params_dict = {k: v.numel() for k, v in m.state_dict().items()}
        to_json_file(n_params_dict, os.path.join(self.output_dir, 'num_model_parameters.json'))

        dev = self.device
        self.logger.info('Device set to {}'.format(dev))
        m.to(dev)

        trainer = self.__get_trainer__()

        # train and validate
        best_val_metrics, best_model, info_training = trainer.train_and_validate(model=m,
                                                                                 train_loader=train_loader,
                                                                                 val_loader=val_loader,
                                                                                 metric_class_list=metric_class_list)

        best_val_metrics_dict = {x.get_name(): x.get_value() for x in best_val_metrics}

        to_json_file(best_val_metrics_dict, os.path.join(self.output_dir, 'best_validation_metrics.json'))
        to_json_file(info_training, os.path.join(self.output_dir, 'info_training.json'))

        if not do_test:
            return best_val_metrics
        else:

            self.__save_test_model_params__(best_model)

            test_loader_dict = dataset_loader.get_test_data_loader_dict()
            test_metrics_dict = {}
            test_prediction_dict = {}
            avg_metrics = []
            for test_name, test_loader in test_loader_dict.items():
                test_metrics, test_prediction = trainer.test(best_model, test_loader,
                                                             metric_class_list=metric_class_list)
                if len(avg_metrics) == 0:
                    avg_metrics = copy.deepcopy(test_metrics)
                else:
                    for i in range(len(avg_metrics)):
                        avg_metrics[i] = avg_metrics[i] + test_metrics[i]

                test_metrics_dict[test_name] = {x.get_name(): x.get_value() for x in test_metrics}
                test_prediction_dict[test_name] = test_prediction

            to_json_file(test_metrics_dict, os.path.join(self.output_dir, 'test_metrics.json'))
            to_torch_file(test_prediction_dict, os.path.join(self.output_dir, 'test_prediction.pth'))

            # the output is printed. average over all test datasets
            for i in range(len(avg_metrics)):
                avg_metrics[i] = avg_metrics[i]/len(test_loader_dict)

            return avg_metrics
