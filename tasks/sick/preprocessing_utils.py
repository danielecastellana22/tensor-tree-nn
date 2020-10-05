import os
import networkx as nx
from tqdm import tqdm
from preprocessing.base import NlpParsedTreesPreprocessor
from preprocessing.utils import ConstValues, load_embeddings
from utils.misc import eprint
from utils.serialization import to_pkl_file, from_pkl_file


class SickParsedTreesPreprocessor(NlpParsedTreesPreprocessor):

    def __init__(self, config):
        super(SickParsedTreesPreprocessor, self).__init__(config)

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir

        tree_type = config.preprocessor_config.tree_type

        # set file names
        file_names = {'train': ['SICK_train_{}.pkl'.format(x) for x in tree_type],
                      'validation': ['SICK_trial_{}.pkl'.format(x) for x in tree_type],
                      'test': ['SICK_test_{}.pkl'.format(x) for x in tree_type]}

        # preprocessing trees
        for tag_name, f_list in file_names.items():
            parsed_trees_list = []
            for f in f_list:
                parsed_trees_list.append(from_pkl_file(os.path.join(input_dir, f)))

            n_trees = len(parsed_trees_list[0])
            parsed_trees = [{'tree_a': tuple([v[i]['tree_a'] for v in parsed_trees_list]),
                             'tree_b': tuple([v[i]['tree_b'] for v in parsed_trees_list]),
                             'relatedness': parsed_trees_list[0][i]['relatedness'],
                             'entailment': parsed_trees_list[0][i]['entailment']} for i in range(n_trees)]

            self.__init_stats__(tag_name)

            data_list = []

            for x in tqdm(parsed_trees, desc='Preprocessing {}'.format(tag_name)):
                t_a = self.tree_transformer.transform(*x['tree_a'])
                t_b = self.tree_transformer.transform(*x['tree_b'])

                self.__assign_node_features__(t_a)
                self.__assign_node_features__(t_b)

                self.__update_stats__(tag_name, t_a)
                self.__update_stats__(tag_name, t_b)

                dgl_t_a = self.__nx_to_dgl__(t_a)
                dgl_t_b = self.__nx_to_dgl__(t_b)
                data_list.append((dgl_t_a, dgl_t_b, x['relatedness'], x['entailment']))

            self.__print_stats__(tag_name)
            to_pkl_file(data_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))

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