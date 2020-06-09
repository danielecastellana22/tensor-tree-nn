from experiments.base import Experiment
import torch as th
import torch.nn.functional as F
import dgl
from treeRNN.models import TreeModel
from tasks.utils.classifiers import OneLayerNN


class TrecExperiment(Experiment):

    def __init__(self, config, output_dir, logger):
        super(TrecExperiment, self).__init__(config, output_dir, logger)

    def __get_num_classes__(self):
        out_type = self.config.dataset_config.output_type
        if out_type == 'coarse':
            return 6
        elif out_type == 'fine':
            return 50
        else:
            raise ValueError('Output type not known!')

    def __create_model__(self):
        tree_model_config = self.config.tree_model_config
        num_classes = self.__get_num_classes__()
        output_model_config = self.config.output_model_config

        h_size = tree_model_config.h_size

        input_module = self.__create_input_embedding_module__()
        type_module = self.__create_type_embedding_module__()
        cell_module = self.__create_cell_module__()
        output_module = OneLayerNN(h_size, num_classes, **output_model_config)

        return TreeModel(input_module, output_module, cell_module, type_module, only_root_state=True)

    def __get_loss_function__(self):
        def f(output_model, true_label):
            return F.cross_entropy(output_model, true_label, reduction='mean')

        return f

    def __get_batcher_function__(self):
        device = self.__get_device__()
        num_classes = self.__get_num_classes__()

        def batcher_dev(tuple_data):
            tree_list, coarse_label_list, fine_label_list = zip(*tuple_data)
            batched_trees = dgl.batch(tree_list)

            batched_trees.to(device)

            if num_classes == 6:
                out = th.LongTensor(coarse_label_list)
                out.to(device)
            else:
                out = th.LongTensor(fine_label_list)
                out.to(device)

            return [batched_trees], out

        return batcher_dev
