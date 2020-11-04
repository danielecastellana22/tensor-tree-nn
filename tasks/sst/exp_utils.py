import os
import networkx as nx
from tqdm import tqdm
from experiments.base import CollateFun
import dgl
from preprocessing.base import NlpParsedTreesPreprocessor
from preprocessing.utils import ConstValues
from utils.misc import eprint
from utils.serialization import from_pkl_file, to_pkl_file


class SstParsedTreesPreprocessor(NlpParsedTreesPreprocessor):

    def __init__(self, config):
        super(SstParsedTreesPreprocessor, self).__init__(config)

    def __get_output_type(self):
        if self.config.output_type == 'fine-grained':
            return 0
        elif self.config.output_type == 'binary':
            return 1
        else:
            raise ValueError('Output type not known!')

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir
        tree_type = config.preprocessor_config.tree_type
        output_type = self.__get_output_type()

        # set file names
        file_names = {'train': ['train_{}.pkl'.format(x) for x in tree_type],
                      'validation': ['validation_{}.pkl'.format(x) for x in tree_type],
                      'test': ['test_{}.pkl'.format(x) for x in tree_type]}

        sentiment_map_file = 'sentiment_map.pkl'

        # load sentiment map
        eprint('Loading sentiment map.')
        sentiment_map = from_pkl_file(os.path.join(input_dir, sentiment_map_file))

        # preprocessing trees
        for tag_name, f_list in file_names.items():
            nx_tree_list = []
            for f in f_list:
                nx_tree_list.append(from_pkl_file(os.path.join(input_dir, f)))
            nx_tree_list = list(zip(*nx_tree_list))

            self.__init_stats__(tag_name)

            tree_list = []

            for x in tqdm(nx_tree_list, desc='Preprocessing {}'.format(tag_name)):
                t = self.tree_transformer.transform(*x)
                if self.__assign_node_features__(t, sentiment_map, output_type):
                    # assign only if there is a label on the root (missing labe means neutral)
                    self.__update_stats__(tag_name, t)
                    tree_list.append(self.__nx_to_dgl__(t))

            self.__print_stats__(tag_name)
            to_pkl_file(tree_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))

        # save all stats
        self.__save_stats__()

        # compute and sabe word_embeddings
        self.__save_word_embeddings__()

    def __assign_node_features__(self, t: nx.DiGraph, *args):

        sentiment_map = args[0]
        output_type = args[1]

        def _rec_assign(node_id, pos):
            all_ch = list(t.predecessors(node_id))

            tokenid_word_list = []
            for p, ch_id in enumerate(all_ch):
                s = _rec_assign(ch_id, p)
                tokenid_word_list += s

            t.nodes[node_id]['pos'] = pos

            if 'word' in t.nodes[node_id]:
                node_word = t.nodes[node_id]['word'].lower()
                tokenid_word_list += [(t.nodes[node_id]['token_id'], node_word)]
                t.nodes[node_id]['x'] = self.__get_word_id__(node_word)
            else:
                t.nodes[node_id]['x'] = ConstValues.NO_ELEMENT

            phrase_key = tuple([x[1] for x in sorted(tokenid_word_list)])
            if phrase_key in sentiment_map:
                sentiment_label = sentiment_map[phrase_key]
                if output_type == 1:
                    # binary classification
                    if sentiment_label < 2:
                         sentiment_label = 0
                    elif sentiment_label == 2:
                        sentiment_label = ConstValues.NO_ELEMENT
                    else:
                        sentiment_label = 1
            else:
                sentiment_label = ConstValues.NO_ELEMENT

            t.nodes[node_id]['y'] = sentiment_label

            # set type
            if self.typed:
                if 'type' in t.nodes[node_id]:
                    tag = t.nodes[node_id]['type']
                    t.nodes[node_id]['t'] = self.__get_type_id__(tag)
                else:
                    t.nodes[node_id]['t'] = ConstValues.NO_ELEMENT

            return tokenid_word_list

        # find the root
        root_list = [x for x in t.nodes if t.out_degree(x) == 0]
        assert len(root_list) == 1
        _rec_assign(root_list[0], -1)

        if t.nodes[root_list[0]]['y'] != -1:
            return True
        else:
            return False


class SstCollateFun(CollateFun):

    def __call__(self, tuple_data):
        batched_trees = dgl.batch(tuple_data)
        batched_trees.to(self.device)
        return [batched_trees], batched_trees.ndata['y']