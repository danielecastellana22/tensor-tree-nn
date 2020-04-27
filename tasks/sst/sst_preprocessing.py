import os
import networkx as nx
from tqdm import tqdm
from preprocessing.base import Preprocessor, NlpParsedTreesPreprocessor
from preprocessing.utils import ConstValues, load_embeddings
from preprocessing.tree_conversions import string_to_nltk_tree, nx_to_dgl, nltk_tree_to_nx
from utils.utils import eprint
from utils.serialization import to_pkl_file, from_pkl_file


class SstBinaryTreesPreprocessor(Preprocessor):

    def __init__(self, config):
        super(SstBinaryTreesPreprocessor, self).__init__(config, False)

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

        # preprocessing binary trees
        for tag_name, f in file_names.items():
            input_file_path = os.path.join(input_dir, f)
            tree_list = []
            self.__init_stats__(tag_name)

            n_words_not_in_vocab = 0
            with open(input_file_path, 'r', encoding='utf-8') as rf:
                for i, l in enumerate(tqdm(rf.readlines(), desc='Reading trees from {}: '.format(f))):
                    t = string_to_nltk_tree(l)
                    bin_t = nltk_tree_to_nx(t, collapsePOS=True,
                                            get_internal_node_dict=lambda label: {'y': int(label)},
                                            get_leaf_node_dict=lambda label: {'word': label.lower().replace('\/', '/')})

                    n_words_not_in_vocab += self.__assign_node_features__(bin_t, words_vocab)
                    self.__update_stats__(tag_name, bin_t)
                    dgl_t = nx_to_dgl(bin_t, node_attrs=['x', 'y'])

                    tree_list.append(dgl_t)

            to_pkl_file(tree_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))
            eprint('{} words not found in the vocabulary.'.format(n_words_not_in_vocab))
            self.__print_stats__(tag_name)

        self.__save_stats__()

        eprint('Loading word embeddings.')
        pretrained_embs = load_embeddings(pretrained_embs_file, words_vocab, embedding_dim=embedding_dim)
        to_pkl_file(pretrained_embs, os.path.join(output_dir, 'pretrained_embs.pkl'))

    @staticmethod
    def __assign_node_features__(t: nx.DiGraph, words_vocab):
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

                if node_word in words_vocab:
                    t.nodes[node_id]['x'] = words_vocab[node_word]
                else:
                    t.nodes[node_id]['x'] = ConstValues.UNK
                    words_not_in_vocab += 1

            else:
                t.nodes[node_id]['x'] = ConstValues.NO_ELEMENT

        # find the root
        root_list = [x for x in t.nodes if t.out_degree(x) == 0]
        assert t.nodes[root_list[0]]['y'] != -1
        return words_not_in_vocab


class SstParsedTreesPreprocessor(NlpParsedTreesPreprocessor):

    def __init__(self, config):
        super(SstParsedTreesPreprocessor, self).__init__(config)

    def preprocess(self):
        config = self.config
        input_dir = config.input_dir
        output_dir = config.output_dir
        tree_type = config.preprocessor_config.tree_type

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
                self.__assign_node_features__(t, sentiment_map)
                self.__update_stats__(tag_name, t)
                tree_list.append(self.__nx_to_dgl__(t))

            self.__print_stats__(tag_name)
            to_pkl_file(tree_list, os.path.join(output_dir, '{}.pkl'.format(tag_name)))

        # save all stats
        self.__save_stats__()

        eprint('Loading word embeddings.')
        pretrained_embs_file = config.pretrained_embs_file
        embedding_dim = config.embedding_dim
        pretrained_embs = load_embeddings(pretrained_embs_file, self.words_vocab, embedding_dim=embedding_dim)
        to_pkl_file(pretrained_embs, os.path.join(output_dir, 'pretrained_embs.pkl'))

        if 'type_pretrained_embs_file' in config:
            eprint('Loading type embeddings.')
            type_pretrained_embs = load_embeddings(config.type_pretrained_embs_file,
                                                   self.types_vocab,
                                                   embedding_dim=config.type_embedding_dim)
            to_pkl_file(type_pretrained_embs, os.path.join(output_dir, 'type_pretrained_embs.pkl'))

    def __assign_node_features__(self, t: nx.DiGraph, sentiment_map):

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
                t.nodes[node_id]['x'] = self.words_vocab[node_word]
            else:
                t.nodes[node_id]['x'] = ConstValues.NO_ELEMENT

            phrase_key = tuple(sorted(list(set(phrase_subtree))))
            if phrase_key in sentiment_map:
                sentiment_label = sentiment_map[phrase_key]
            else:
                sentiment_label = ConstValues.NO_ELEMENT

            t.nodes[node_id]['y'] = sentiment_label

            # set type
            if self.typed:
                if 'type' in t.nodes[node_id]:
                    tag = t.nodes[node_id]['type']
                    t.nodes[node_id]['type_id'] = self.__get_type_id__(tag)
                else:
                    t.nodes[node_id]['type_id'] = ConstValues.NO_ELEMENT

                return phrase_subtree

        # find the root
        root_list = [x for x in t.nodes if t.out_degree(x) == 0]
        assert len(root_list) == 1
        _rec_assign(root_list[0])
        assert t.nodes[root_list[0]]['y'] != -1