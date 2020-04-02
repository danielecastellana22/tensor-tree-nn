import os
import networkx as nx
from tqdm import tqdm
from preprocessing.base import Preprocessor
from preprocessing.utils import ConstValues, load_embeddings
from preprocessing.tree_conversions import string_to_nltk_tree, nx_to_dgl, nltk_tree_to_nx
from utils.utils import eprint, string2class
from utils.serialization import to_pkl_file, from_pkl_file, to_json_file
from preprocessing.NLP.transformers import CombinedTreeTransformer


class SstBinaryTreesPreprocessor(Preprocessor):

    def __init__(self, config):
        super(SstBinaryTreesPreprocessor, self).__init__(config)
        preprocessor_config = self.config.preprocessor_config
        self.type = preprocessor_config.type
        if self.type == 'None':
            self.tree_transformer = None
        else:
            self.type_stats = {}
            self.tree_transformer = CombinedTreeTransformer(self.type)
            self.parsed_trees_dir = preprocessor_config.parsed_trees_dir

    def __init_stats__(self, tag_name):
        super(SstBinaryTreesPreprocessor, self).__init_stats__(tag_name)
        if self.type != 'None':
            self.stats[tag_name]['no_types'] = 0
            self.type_stats[tag_name] = {}

    def __update_stats__(self, tag_name, t:nx.DiGraph):
        super(SstBinaryTreesPreprocessor, self).__update_stats__(tag_name, t)
        if self.type != 'None':
            self.stats[tag_name]['no_types'] += len([i for i, d in t.nodes(data=True) if d['type_id'] == ConstValues.NO_ELEMENT])
            for i, d in t.nodes(data=True):
                if 'type' in d:
                    tt = d['type']
                    if tt not in self.type_stats[tag_name]:
                        self.type_stats[tag_name][tt] = 0
                    self.type_stats[tag_name][tt] += 1

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir
        file_names = {'train': 'train.txt',
                      'validation': 'validation.txt',
                      'test': 'test.txt'}
        words_vocab_file = 'vocab.txt'
        pretrained_embs_file = config.pretrained_embs_file
        embedding_dim = config.embedding_dim

        # load vocabulary
        eprint('Loading word vocabulary.')
        words_vocab = {'unk': 0}
        with open(os.path.join(input_dir, words_vocab_file), encoding='utf-8') as f:
            for l in tqdm(f.readlines(), desc='Creating words vocabulary'):
                l = l.strip().lower().replace('\/', '/')
                words_vocab[l] = len(words_vocab)

        if self.type != 'None':
            types_vocab = {}
        else:
            types_vocab = None

        # preprocessing binary trees
        for tag_name, f in file_names.items():
            input_file_path = os.path.join(input_dir, f)
            tree_list = []
            self.__init_stats__(tag_name)

            if self.type == 'const_tag':
                self.parsed_trees = from_pkl_file(os.path.join(self.parsed_trees_dir, '{}_const.pkl'.format(tag_name)))
            elif self.type == 'dep_rel':
                self.parsed_trees = from_pkl_file(os.path.join(self.parsed_trees_dir, '{}_dep.pkl'.format(tag_name)))

            n_words_not_in_vocab = 0
            with open(input_file_path, 'r', encoding='utf-8') as rf:
                for i, l in enumerate(tqdm(rf.readlines(), desc='Reading trees from {}: '.format(f))):
                    t = string_to_nltk_tree(l)
                    bin_t = nltk_tree_to_nx(t, collapsePOS=True,
                                            get_internal_node_dict=lambda label: {'y': int(label)},
                                            get_leaf_node_dict=lambda label: {'word': label})

                    if self.tree_transformer is not None:
                        bin_t = self.tree_transformer.transform(bin_t, self.parsed_trees[i])

                    n_words_not_in_vocab += self.__assign_node_features__(bin_t, words_vocab, types_vocab)
                    self.__update_stats__(tag_name, bin_t)
                    if self.type == 'None':
                        dgl_t = nx_to_dgl(bin_t, node_attrs=['x', 'y'])
                    else:
                        dgl_t = nx_to_dgl(bin_t, node_attrs=['x', 'y', 'type_id'])
                    tree_list.append(dgl_t)

            to_pkl_file(tree_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))
            eprint('{} words not found in the vocabulary.'.format(n_words_not_in_vocab))
            self.__print_stats__(tag_name)

        # store types_vocab
        if self.type != 'None':
            self.stats['num_types'] = len(types_vocab)
            eprint('{} types found.'.format(len(types_vocab)))
            eprint('Saving types vocab.')
            to_json_file(types_vocab, os.path.join(output_dir, 'types_vocab.json'))
            eprint('Saving types stats.')
            to_json_file(self.type_stats, os.path.join(output_dir, 'type_stats.json'))

        # save all stats
        eprint('Saving stats.')
        to_json_file(self.stats, os.path.join(output_dir, 'stats.json'))


        eprint('Loading word embeddings.')
        pretrained_embs = load_embeddings(pretrained_embs_file, words_vocab, embedding_dim=embedding_dim)
        to_pkl_file(pretrained_embs, os.path.join(output_dir, 'pretrained_embs.pkl'))

    @staticmethod
    def __assign_node_features__(t: nx.DiGraph, words_vocab, types_vocab):
        words_not_in_vocab = 0

        def _rec_assign(node_id):
            nonlocal words_not_in_vocab
            assert len(list(t.successors(node_id))) <= 1
            all_ch = list(t.predecessors(node_id))

            for ch_id in all_ch:
                _rec_assign(ch_id)

            # set word
            if 'word' in t.nodes[node_id]:
                node_word = t.nodes[node_id]['word']
                node_word = node_word.lower().replace('\/', '/')

                if node_word in words_vocab:
                    t.nodes[node_id]['x'] = words_vocab[node_word]
                else:
                    t.nodes[node_id]['x'] = ConstValues.UNK
                    words_not_in_vocab += 1

            else:
                t.nodes[node_id]['x'] = ConstValues.NO_ELEMENT

            # set type
            if types_vocab is not None:
                if 'type' in t.nodes[node_id]:
                    tag = t.nodes[node_id]['type']
                    if tag not in types_vocab:
                        # update types vocab
                        types_vocab[tag] = len(types_vocab)
                    t.nodes[node_id]['type_id'] = types_vocab[tag]
                else:
                    t.nodes[node_id]['type_id'] = ConstValues.NO_ELEMENT

        # find the root
        root_list = [x for x in t.nodes if t.out_degree(x) == 0]
        assert len(root_list) == 1
        _rec_assign(root_list[0])
        assert t.nodes[root_list[0]]['y'] != -1
        return words_not_in_vocab


class SstParsedTreesPreprocessor(Preprocessor):

    def __init__(self, config):
        super(SstParsedTreesPreprocessor, self).__init__(config)
        preprocessor_config = self.config.preprocessor_config
        self.tree_type = preprocessor_config.tree_type
        # create tree transformer
        tree_transformer_class = string2class(preprocessor_config.tree_transformer_class)
        if 'tree_transformer_params' in preprocessor_config:
            self.tree_transformer = tree_transformer_class(**preprocessor_config.tree_transformer_params)
        else:
            self.tree_transformer = tree_transformer_class()

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir

        # set file names
        file_names = {'train': 'train_{}.pkl'.format(self.tree_type),
                      'validation': 'validation_{}.pkl'.format(self.tree_type),
                      'test': 'test_{}.pkl'.format(self.tree_type)}

        words_vocab_file = 'vocab.pkl'
        sentiment_map_file = 'sentiment_map.pkl'
        pretrained_embs_file = config.pretrained_embs_file
        embedding_dim = config.embedding_dim

        # load vocabulary
        eprint('Loading word vocabulary.')
        words_vocab = from_pkl_file(os.path.join(input_dir, words_vocab_file))

        # load sentiment map
        eprint('Loading sentiment map.')
        sentiment_map = from_pkl_file(os.path.join(input_dir, sentiment_map_file))

        # preprocessing trees
        for tag_name, f in file_names.items():
            input_file_path = os.path.join(input_dir, f)
            all_tree_list = from_pkl_file(input_file_path)
            self.__init_stats__(tag_name)
            # extract dep tree
            tree_list = []
            for x in tqdm(all_tree_list, desc='Preprocessing {}'.format(f)):
                t = self.tree_transformer.transform(x)
                self.__assign_node_features__(t, words_vocab, sentiment_map)
                self.__update_stats__(tag_name, t)
                tree_list.append(nx_to_dgl(t, node_attrs=['x', 'y']))

            self.__print_stats__(tag_name)
            to_pkl_file(tree_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))

        # save all stats
        to_json_file(self.stats, os.path.join(output_dir, 'stats.json'))

        eprint('Loading word embeddings.')
        pretrained_embs = load_embeddings(pretrained_embs_file, words_vocab, embedding_dim=embedding_dim)
        to_pkl_file(pretrained_embs, os.path.join(output_dir, 'pretrained_embs.pkl'))

    @staticmethod
    def __assign_node_features__(t:nx.DiGraph, words_vocab, sentiment_map):

        def _rec_assign(node_id):
            assert len(list(t.successors(node_id))) <= 1
            all_ch = list(t.predecessors(node_id))

            phrase_subtree = []
            for ch_id in all_ch:
                s = _rec_assign(ch_id)
                phrase_subtree += s

            if 'word' in t.nodes[node_id]:
                node_word = t.nodes[node_id]['word'].lower()
                phrase_subtree += [node_word]
                t.nodes[node_id]['x'] = words_vocab[node_word]
            else:
                t.nodes[node_id]['x'] = ConstValues.NO_ELEMENT

            phrase_key = tuple(sorted(list(set(phrase_subtree))))
            if phrase_key in sentiment_map:
                sentiment_label = sentiment_map[phrase_key]
            else:
                sentiment_label = ConstValues.NO_ELEMENT

            t.nodes[node_id]['y'] = sentiment_label

            return phrase_subtree

        # find the root
        root_list = [x for x in t.nodes if t.out_degree(x) == 0]
        assert len(root_list) == 1
        _rec_assign(root_list[0])
        assert t.nodes[root_list[0]]['y'] != -1
