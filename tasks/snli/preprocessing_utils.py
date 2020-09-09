import os
import networkx as nx
from tqdm import tqdm
from preprocessing.base import NlpParsedTreesPreprocessor
from preprocessing.utils import ConstValues, load_embeddings
from utils.utils import eprint
from utils.serialization import to_pkl_file, from_pkl_file


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

    def __assign_node_features__(self, t: nx.DiGraph):

        def _rec_assign(node_id):
            #assert len(list(t.successors(node_id))) <= 1
            all_ch = list(t.predecessors(node_id))

            phrase_subtree = []
            for ch_id in all_ch:
                _rec_assign(ch_id)

            t.nodes[node_id]['y'] = ConstValues.NO_ELEMENT

            if 'word' in t.nodes[node_id]:
                node_word = t.nodes[node_id]['word'].lower()
                phrase_subtree += [node_word]
                t.nodes[node_id]['x'] = self.__get_word_id__(node_word)
                t.nodes[node_id]['x_mask'] = 1
            else:
                t.nodes[node_id]['x'] = ConstValues.NO_ELEMENT
                t.nodes[node_id]['x_mask'] = 0

            # set type
            if self.typed:
                if 'type' in t.nodes[node_id]:
                    tag = t.nodes[node_id]['type']
                    t.nodes[node_id]['t'] = self.__get_type_id__(tag)
                    t.nodes[node_id]['t_mask'] = 1
                else:
                    t.nodes[node_id]['t'] = ConstValues.NO_ELEMENT
                    t.nodes[node_id]['t_mask'] = 0

        # find the root
        root_list = [x for x in t.nodes if t.out_degree(x) == 0]
        assert len(root_list) == 1
        _rec_assign(root_list[0])