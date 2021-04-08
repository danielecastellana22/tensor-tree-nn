import os
from tqdm import tqdm
from exputils.datasets import CollateFun
import torch as th
import dgl
from preprocessing.preprocessors import NlpParsedTreesPreprocessor
from exputils.serialisation import from_pkl_file, to_pkl_file


class TrecParsedTreesPreprocessor(NlpParsedTreesPreprocessor):

    def __init__(self, config):
        super(TrecParsedTreesPreprocessor, self).__init__(config)

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir

        tree_type = config.preprocessor_config.tree_type

        # set file names
        file_names = {'train': ['train_{}.pkl'.format(x) for x in tree_type],
                      'validation': ['validation_{}.pkl'.format(x) for x in tree_type],
                      'test': ['test_{}.pkl'.format(x) for x in tree_type]}

        # preprocessing trees
        for tag_name, f_list in file_names.items():
            parsed_trees_list = []
            for f in f_list:
                parsed_trees_list.append(from_pkl_file(os.path.join(input_dir, f)))

            n_trees = len(parsed_trees_list[0])
            parsed_trees = [{'tree': tuple([v[i]['tree'] for v in parsed_trees_list]),
                             'coarse_label': parsed_trees_list[0][i]['coarse_label'],
                             'fine_label': parsed_trees_list[0][i]['fine_label']} for i in range(n_trees)]

            self.__init_stats__(tag_name)

            data_list = []

            for x in tqdm(parsed_trees, desc='Preprocessing {}'.format(tag_name)):
                t = self.tree_transformer.transform(*x['tree'])

                self.__assign_node_features__(t)

                self.__update_stats__(tag_name, t)

                dgl_t = self.__nx_to_dgl__(t)
                data_list.append((dgl_t, x['coarse_label'], x['fine_label']))

            self.__print_stats__(tag_name)
            to_pkl_file(data_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))

        # save all stats
        self.__save_stats__()
        self.__save_word_embeddings__()


class TrecCollateFun(CollateFun):

    def __init__(self, device, output_type):
        super(TrecCollateFun, self).__init__(device)
        if output_type == 'coarse':
            self.num_classes = 6
        elif output_type == 'fine':
            self.num_classes = 50
        else:
            raise ValueError('Output type not known!')

    def __call__(self, tuple_data):
        tree_list, coarse_label_list, fine_label_list = zip(*tuple_data)
        batched_trees = dgl.batch(tree_list)
        batched_trees.to(self.device)

        if self.num_classes == 6:
            out = th.tensor(coarse_label_list, dtype=th.long)
            out.to(self.device)
        else:
            out = th.tensor(fine_label_list, dtype=th.long)
            out.to(self.device)

        return [batched_trees], out