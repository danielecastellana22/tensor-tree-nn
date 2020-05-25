from abc import abstractmethod
import os
from utils.utils import set_initial_seed, string2class
from utils.serialization import to_json_file, from_json_file, from_pkl_file, to_torch_file, from_torch_file
import torch as th
from preprocessing.dataset import ListDataset
from treeRNN.cells import TypedTreeCell
from training.trainers import BasicTrainer
import torch.nn as nn

# base class for all experiments
class Experiment:

    def __init__(self, config, output_dir, logger):

        self.config = config
        self.output_dir = output_dir
        self.logger = logger

        # save config
        to_json_file(self.config, os.path.join(output_dir, 'config.json'))

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

    @staticmethod
    def __create_embedding_module__(emb_module_config):
        embedding_type =  emb_module_config.embedding_type
        if embedding_type == 'pretrained':
            np_array = from_pkl_file(emb_module_config.pretrained_embs)
            return nn.Embedding.from_pretrained(th.tensor(np_array, dtype=th.float), freeze=emb_module_config.freeze)
        elif embedding_type == 'one_hot':
            num_embs = emb_module_config.num_embs
            return nn.Embedding.from_pretrained(th.eye(num_embs, num_embs), freeze=emb_module_config.freeze)
        elif embedding_type == 'random':
            num_embs = emb_module_config.num_embs
            emb_size = emb_module_config.emb_size
            return nn.Embedding(num_embs, emb_size)
        else:
            raise ValueError('Embedding type is unkown!')

    def __create_input_embedding_module__(self):
        if 'input_model_config' in self.config:
            return self.__create_embedding_module__(self.config.input_model_config)
        else:
            return None

    def __create_type_embedding_module__(self):
        if 'type_model_config' in self.config:
            return self.__create_embedding_module__(self.config.type_model_config)
        else:
            return None

    @abstractmethod
    def __create_model__(self):
        raise NotImplementedError('This methos must be define in a sublclass!')

    def __get_optimiser__(self, model):
        optim_config = self.config.optimiser_config
        optim_class = string2class(optim_config.optimiser_class)
        params_groups = dict(optim_config.optimiser_params) if 'optimiser_params' in optim_config else {}
        params_groups.update({'params': list(model.parameters())})

        return optim_class([params_groups])

    @abstractmethod
    def __get_loss_function__(self):
        raise NotImplementedError('This methos must be define in a sublclass!')

    @abstractmethod
    def __get_batcher_function__(self):
        raise NotImplementedError('This methos must be define in a sublclass!')

    def __save_test_model_params__(self, best_model):
        to_torch_file(best_model.state_dict(), os.path.join(self.output_dir, 'params_learned.pth'))

    def __create_cell_module__(self):
        cell_config = self.config.tree_model_config.cell_config
        h_size = self.config.tree_model_config.h_size
        if 'input_model_config' in self.config:
            x_size = self.config.input_model_config.emb_size
        else:
            x_size = self.config.tree_model_config.x_size

        cell_class = string2class(cell_config.cell_class)
        num_types = cell_config.num_types if hasattr(cell_config, 'num_types') else None
        cell_params = dict(cell_config.cell_params)
        cell_params['aggregator_class'] = string2class(cell_params['aggregator_class'])

        if 'type_model_config' in self.config:
            cell_params['type_emb_size'] = self.config.type_model_config.emb_size

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

    def __get_device__(self):
        dev = self.config.others_config.gpu
        cuda = dev >= 0
        device = th.device('cuda:{}'.format(dev)) if cuda else th.device('cpu')
        if cuda:
            th.cuda.set_device(dev)

        return device

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

        m = self.__create_model__()
        # save number of parameters
        n_params_dict = {k: v.numel() for k, v in m.state_dict().items()}
        to_json_file(n_params_dict, os.path.join(self.output_dir, 'num_model_parameters.json'))

        opt = self.__get_optimiser__(m)

        # train and validate
        best_val_metrics, best_model, info_training = BasicTrainer.train_and_validate(m, self.__get_loss_function__(), opt, trainset, valset,
                                                                                      batcher_fun=self.__get_batcher_function__(),
                                                                                      logger=self.logger.getChild('train'),
                                                                                      metric_class_list=metric_class_list,
                                                                                      **training_config)

        best_val_metrics_dict = {x.get_name(): x.get_value() for x in best_val_metrics}

        to_json_file(best_val_metrics_dict, os.path.join(self.output_dir, 'best_validation_metrics.json'))
        to_json_file(info_training, os.path.join(self.output_dir, 'info_training.json'))

        if not do_test:
            return best_val_metrics
        else:

            self.__save_test_model_params__(best_model)

            testset = self.__load_test_data__()

            test_metrics, test_prediction = BasicTrainer.test(best_model, testset,
                                                              batcher_fun=self.__get_batcher_function__(),
                                                              logger=self.logger.getChild('test'),
                                                              metric_class_list=metric_class_list,
                                                              batch_size=training_config.batch_size)

            test_metrics_dict = {x.get_name(): x.get_value() for x in test_metrics}
            to_json_file(test_metrics_dict, os.path.join(self.output_dir, 'test_metrics.json'))
            to_torch_file(test_prediction, os.path.join(self.output_dir, 'test_prediction.pth'))
            return test_metrics
