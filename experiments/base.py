from abc import abstractmethod, ABC
import os
from utils.utils import set_initial_seed, string2class
from utils.serialization import to_json_file, from_json_file, from_pkl_file, to_torch_file, from_torch_file
import torch as th
from preprocessing.dataset import ListDataset
from treeRNN.cells import TypedTreeCell
from training.trainers import NeuralTrainer, EMTrainer
from treeRNN.models import TreeModel
from HTMM.models import BHTMM


# base class for all experiments
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
    @staticmethod
    def __create_module__(config, **other_params):
        class_name = string2class(config['class'])
        params: dict = config['params']
        params.update(other_params)
        return class_name(**params)

    @abstractmethod
    def __create_tree_module__(self):
        raise NotImplementedError('This methos must be define in a sublclass!')

    ####################################################################################################################
    # TRAINER FUNCTIONS
    ####################################################################################################################
    @abstractmethod
    def __get_trainer__(self):
        raise NotImplementedError('This methos must be define in a sublclass!')

    @abstractmethod
    def __get_trainer_arguments__(self, m):
        raise NotImplementedError('This methos must be define in a sublclass!')

    @abstractmethod
    def __get_batcher_function__(self):
        raise NotImplementedError('This methos must be define in a sublclass!')

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
        training_config = self.config.training_config

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

        trainer_arg = self.__get_trainer_arguments__(m)
        trainer = self.__get_trainer__()

        # train and validate
        best_val_metrics, best_model, info_training = trainer.train_and_validate(model=m,
                                                                                 trainset=trainset,
                                                                                 valset=valset,
                                                                                 batcher_fun=self.__get_batcher_function__(),
                                                                                 metric_class_list=metric_class_list,
                                                                                 **training_config,
                                                                                 **trainer_arg)

        best_val_metrics_dict = {x.get_name(): x.get_value() for x in best_val_metrics}

        to_json_file(best_val_metrics_dict, os.path.join(self.output_dir, 'best_validation_metrics.json'))
        to_json_file(info_training, os.path.join(self.output_dir, 'info_training.json'))

        if not do_test:
            return best_val_metrics
        else:

            self.__save_test_model_params__(best_model)

            testset = self.__load_test_data__()

            test_metrics, test_prediction = trainer.test(best_model, testset,
                                                         batcher_fun=self.__get_batcher_function__(),
                                                         metric_class_list=metric_class_list,
                                                         batch_size=training_config.batch_size)

            test_metrics_dict = {x.get_name(): x.get_value() for x in test_metrics}
            to_json_file(test_metrics_dict, os.path.join(self.output_dir, 'test_metrics.json'))
            to_torch_file(test_prediction, os.path.join(self.output_dir, 'test_prediction.pth'))
            return test_metrics


# TODO: rewrite this
class NeuralExperiment(Experiment):

    def __get_trainer__(self):
        return NeuralTrainer(self.debug_mode, self.logger)

    def __get_trainer_arguments__(self, m):
        opt = self.__get_optimiser__(m)
        loss_f = self.__get_loss_function__()
        return {'loss_function': loss_f, 'optimiser': opt}

    def __get_optimiser__(self, model):
        optim_config = self.config.optimiser_config
        optim_class = string2class(optim_config.optimiser_class)
        params_groups = dict(optim_config.optimiser_params) if 'optimiser_params' in optim_config else {}
        params_groups.update({'params': list(model.parameters())})

        return optim_class([params_groups])

    @abstractmethod
    def __get_loss_function__(self):
        raise NotImplementedError('This methos must be define in a sublclass!')

    # TODO: rewrite this
    def __create_cell_module__(self):
        cell_config = self.config.tree_module_config.cell_config
        h_size = self.config.tree_module_config.h_size
        if 'input_module_config' in self.config:
            x_size = self.config.input_module_config.emb_size
        else:
            x_size = self.config.tree_module_config.x_size

        cell_class = string2class(cell_config.cell_class)
        num_types = cell_config.num_types if hasattr(cell_config, 'num_types') else None
        cell_params = dict(cell_config.cell_params)
        cell_params['aggregator_class'] = string2class(cell_params['aggregator_class'])

        if 'type_module_config' in self.config:
            cell_params['type_emb_size'] = self.config.type_module_config.emb_size

        if num_types is None:
            cell = cell_class(x_size=x_size,
                              h_size=h_size,
                              **cell_params)
        else:
            # the key can be used to assign particular
            cell = TypedTreeCell(x_size=x_size,
                                 h_size=h_size,
                                 cell_class=cell_class,
                                 cells_params_list=[cell_params for i in range(num_types)],
                                 share_input_matrices=cell_config.share_input_matrices)

        return cell

    def __create_tree_module__(self):
        tree_module_config = self.config.tree_module_config

        input_module = self.__create_module__(self.config.input_module_config) if 'input_module_config' in self.config else None
        type_module = self.__create_module__(self.config.type_module_config) if 'type_module_config' in self.config else None
        output_module = self.__create_module__(self.config.output_module_config) if 'output_module_config' in self.config else None
        cell_module = self.__create_cell_module__()

        return TreeModel(input_module, output_module, cell_module, type_module, only_root_state=tree_module_config.only_root_state)


class ProbExperiment(Experiment, ABC):

    def __get_trainer__(self):
        return EMTrainer(self.debug_mode, self.logger)

    def __get_trainer_arguments__(self, m):
        return {}

    def __create_tree_module__(self):
        tree_module_config = self.config.tree_module_config
        h_size = tree_module_config.h_size

        x_emission_module = self.__create_module__(self.config.x_emission_config, h_size=h_size)
        y_emission_module = self.__create_module__(self.config.y_emission_config, h_size=h_size)
        state_transition_module = self.__create_module__(tree_module_config.state_transition_config, h_size=h_size)

        return BHTMM(x_emission=x_emission_module,
                     y_emission=y_emission_module,
                     state_transition=state_transition_module,
                     only_root_state=tree_module_config.only_root_state)