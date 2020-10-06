from tqdm import tqdm
from experiments.base import CollateFun
import torch as th
import dgl
from preprocessing.base import NlpParsedTreesPreprocessor
from utils.misc import eprint
from utils.serialization import from_pkl_file, to_pkl_file
import os


class SnliParsedTreesPreprocessor(NlpParsedTreesPreprocessor):

    def __init__(self, config):
        super(SnliParsedTreesPreprocessor, self).__init__(config)

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir

        tree_type = config.preprocessor_config.tree_type

        if len(tree_type) > 1:
            raise ValueError('Only one tree type can be specified!')

        # set file names
        file_names = {'train': ['snli_1.0_train_{}_{}.pkl'.format(i, tree_type[0]) for i in range(12)],
                      'validation': ['snli_1.0_dev_0_{}.pkl'.format(tree_type[0])],
                      'test': ['snli_1.0_test_0_{}.pkl'.format(tree_type[0])]}

        # preprocessing trees
        for tag_name, f_list in file_names.items():

            self.__init_stats__(tag_name)

            for i, f in enumerate(f_list):

                if os.path.exists(os.path.join(output_dir, '{}_{}.pkl'.format(tag_name, i))):
                    eprint("{} already preprocessed!".format(os.path.join(output_dir, '{}_{}.pkl'.format(tag_name, i))))
                    continue

                parsed_trees = from_pkl_file(os.path.join(input_dir, f))

                data_list = []
                for x in tqdm(parsed_trees, desc='Preprocessing {}'.format(tag_name)):
                    t_a = self.tree_transformer.transform(x['tree_a'])
                    t_b = self.tree_transformer.transform(x['tree_b'])

                    self.__assign_node_features__(t_a)
                    self.__assign_node_features__(t_b)

                    self.__update_stats__(tag_name, t_a)
                    self.__update_stats__(tag_name, t_b)

                    dgl_t_a = self.__nx_to_dgl__(t_a)
                    dgl_t_b = self.__nx_to_dgl__(t_b)
                    data_list.append((dgl_t_a, dgl_t_b, x['entailment']))

                to_pkl_file(data_list, os.path.join(output_dir, '{}_{}.pkl'.format(tag_name, i)))

            self.__print_stats__(tag_name)

        # save all stats
        self.__save_stats__()

        self.__save_word_embeddings__()


class SnliCollateFun(CollateFun):

    def __call__(self, tuple_data):
        a_tree_list, b_tree_list, entailment_list = zip(*tuple_data)
        batched_a_trees = dgl.batch(a_tree_list)
        batched_b_trees = dgl.batch(b_tree_list)

        batched_a_trees.to(self.device)
        batched_b_trees.to(self.device)

        out = th.tensor(entailment_list, dtype=th.long)
        out.to(self.device)

        return (batched_a_trees, batched_b_trees), out