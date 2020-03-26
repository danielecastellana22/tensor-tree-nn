import networkx as nx
import dgl
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import namedtuple
import pickle
import os


class TreeDataset(Dataset):

    NODES_ATTRIBUTE_LIST = ['x', 'y', 'mask', 'type_id']
    UNK = 0   # out-of-vocabulary element
    NO_ELEMENT = -1  # flag to indicate missing element

    def __init__(self, path_dir, file_name_list, name, words_vocab=None, types_vocab=None):
        Dataset.__init__(self)
        self.data = []
        self.path_dir = path_dir
        self.file_name_list = file_name_list
        self.name = name

        self.logger = get_sub_logger(self.name)

        self.__init_vocab__('words', words_vocab)
        self.__init_vocab__('types', types_vocab)

        self.max_out_degree_types = {}
        self.max_out_degree = 0

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __init_vocab__(self, vocab_name, vocab_init):
        vocab_attr_name = '{}_vocab'.format(vocab_name)
        flag_attr_name = 'build_{}_vocab'.format(vocab_name)
        counting_attr_name = '{}_counting'.format(vocab_name)

        if vocab_init is None:
            # build the vocabulary
            self.__setattr__(vocab_attr_name, {'unk': SentenceTreeDataset.UNK})
            self.__setattr__(flag_attr_name, True)
        else:
            self.__setattr__(vocab_attr_name, vocab_init)
            self.__setattr__(flag_attr_name, False)

        self.__setattr__(counting_attr_name, {'unk': 0})

    def __search_and_update_vocab__(self, vocab_name, el):
        vocab_attr_name = '{}_vocab'.format(vocab_name)
        flag_attr_name = 'build_{}_vocab'.format(vocab_name)
        counting_attr_name = '{}_counting'.format(vocab_name)

        vocab = self.__getattribute__(vocab_attr_name)
        flag = self.__getattribute__(flag_attr_name)
        counting = self.__getattribute__(counting_attr_name)

        el = el.lower()
        idx = vocab.get(el, SentenceTreeDataset.UNK)

        if flag and idx == SentenceTreeDataset.UNK:
            vocab[el] = len(vocab)
            idx = vocab[el]

        if idx == SentenceTreeDataset.UNK:
            counting['unk'] += 1
        else:
            if el not in counting:
                counting[el] = 0
            counting[el] += 1

        return idx

    def __get_word_id__(self, w):
        return self.__search_and_update_vocab__('words', w)

    def __get_type_id__(self, t):
        return self.__search_and_update_vocab__('types', t)

    @property
    def num_words(self):
        return len(self.__getattribute__('words_vocab'))

    @property
    def num_types(self):
        return len(self.__getattribute__('types_vocab'))

    # parse file and poupulate self.data
    @abstractmethod
    def __parse_file__(self, f_name):
        raise NotImplementedError

    def __load_trees__(self):

        for f_name in self.file_name_list:
            self.logger.info('Loading trees from {}.'.format(f_name))
            self.__parse_file__(os.path.join(self.path_dir, f_name))

        if self.build_types_vocab and self.num_types > 1:  # only one type means there are no types
            self.__filter_types__()

        self.__print_stats__()

    def __to_dgl__(self, nx_t):
        g = dgl.DGLGraph()
        if 'pos' in list(nx_t.edges(data=True))[0][2]:
            g.from_networkx(nx_t, node_attrs=self.NODES_ATTRIBUTE_LIST, edge_attrs=['pos'])
        else:
            g.from_networkx(nx_t, node_attrs=self.NODES_ATTRIBUTE_LIST)

        # compute max_out_degree
        for i in range(g.number_of_nodes()):
            t_id = g.ndata['type_id'][i].item()
            in_deg = g.in_degree(i)
            if t_id not in self.max_out_degree_types:
                self.max_out_degree_types[t_id] = 0
            self.max_out_degree_types[t_id] = max(self.max_out_degree_types[t_id], in_deg)
            self.max_out_degree = max(self.max_out_degree, in_deg)

        return g

    def __print_stats__(self):
        self.logger.info('{} elements loaded.'.format(len(self)))
        self.logger.info('{} different words.'.format(self.num_words))
        self.logger.info('{} different types.'.format(self.num_types))

    @staticmethod
    def __filter_type_single_tree__(t, types_rev_vocab, new_types_vocab, new_types_counting, new_max_out_degree_types):
        for i in range(t.number_of_nodes()):
            t_id = t.ndata['type_id'][i].item()
            t_name = types_rev_vocab[t_id]

            if t_name in new_types_vocab:
                new_t_id = new_types_vocab[t_name]
                new_t_name = t_name
            else:
                new_t_id = DependencyTreeDataset.UNK
                new_t_name = 'unk'

            t.ndata['type_id'][i] = new_t_id
            new_types_counting[new_t_name] += 1

            in_dg = t.in_degree(i)
            new_max_out_degree_types[new_t_name] = max(new_max_out_degree_types[new_t_name], in_dg)

    def __compute_retained_types__(self):
        min_occ = 50
        self.logger.info('Remove types that not occur more than {} times.'.format(min_occ))
        retained_types = [k for k, v in self.types_counting.items() if v >= min_occ and k != 'unk']
        retained_types.insert(0, 'unk')
        return retained_types

    def __filter_types__(self):

        retained_types = self.__compute_retained_types__()
        # new vocab which preserves order
        new_types_vocab = {k: v for v, k in enumerate(retained_types)}
        new_types_counting = {k: 0 for k in retained_types}
        new_max_out_degree_types = {k: 0 for k in retained_types}
        # filter types
        types_rev_vocab = list(self.types_vocab.keys())
        for el in self.data:
            if isinstance(el, tuple):
                for t in el:
                    if isinstance(t, dgl.DGLGraph):
                        self.__filter_type_single_tree__(t, types_rev_vocab, new_types_vocab, new_types_counting, new_max_out_degree_types)
            else:
                assert isinstance(el, dgl.DGLGraph)
                self.__filter_type_single_tree__(el, types_rev_vocab, new_types_vocab, new_types_counting, new_max_out_degree_types)

        self.max_out_degree = max(list(new_max_out_degree_types.values()))
        self.max_out_degree_types = new_max_out_degree_types
        self.types_vocab = new_types_vocab
        self.types_counting = new_types_counting

    # TODO: maybe this should be an external function in the utils file. Hence, we can use ConcatDataset
    @abstractmethod
    def get_loader(self, batch_size, device, shuffle=False):
        raise NotImplementedError


# this class add tag vocab and assume nx graph are in pkl file
class SentenceTreeDataset(TreeDataset):

    NODES_ATTRIBUTE_LIST = TreeDataset.NODES_ATTRIBUTE_LIST + ['tag_id']

    def __init__(self, path_dir, file_name_list, name, words_vocab=None, types_vocab=None, tags_vocab=None):
        TreeDataset.__init__(self, path_dir, file_name_list, name, words_vocab, types_vocab)
        self.__init_vocab__('tags', tags_vocab)

    @property
    def num_tags(self):
        return len(self.__getattribute__('tags_vocab'))

    def __get_tag_id__(self, t):
        return self.__search_and_update_vocab__('tags', t)



class SSTDependencyTreeDataset(DependencyTreeDataset):

    def __init__(self, path_dir, file_name_list, name, do_augmentation='None',
                 words_vocab=None, types_vocab=None, tags_vocab=None):
        DependencyTreeDataset.__init__(self, path_dir, file_name_list, name, do_augmentation,
                                       words_vocab, types_vocab, tags_vocab)
        self.tot_nodes = 0
        self.tot_leaves = 0
        self.no_label = 0
        self.num_classes = 5
        self.__load_sentiment_mapping__()
        self.__load_trees__()

    def __get_sentiment__(self, word_list):
        key = tuple(sorted(word_list))
        if key in self.phrase2sentiment:
            return self.phrase2sentiment[key]
        else:
            return self.NO_ELEMENT

    def __load_sentiment_mapping__(self):
        fname_sentiment_map = 'sentiment_map.txt'

        self.phrase2sentiment = {}
        with open(os.path.join(self.path_dir, fname_sentiment_map), 'r', encoding='utf-8') as f_sentiment_map:
            for l in tqdm(f_sentiment_map.readlines(), desc='Loading sentiment labels: '):
                v = l.split('|')
                key = tuple(sorted(v[0].split(' ')))

                self.phrase2sentiment[key] = int(v[1])

    def __compute_node_features__(self, dep_tree: nx.DiGraph):

        def _rec_visit_nx(node_id):
            phrase_subtree = []
            all_ch = list(dep_tree.predecessors(node_id))

            for ch_id in all_ch:
                s = _rec_visit_nx(ch_id)
                phrase_subtree += s

            if 'word' in dep_tree.nodes[node_id]:
                node_word = dep_tree.nodes[node_id]['word']
                phrase_subtree += [node_word]
                dep_tree.nodes[node_id]['x'] = self.__get_word_id__(node_word)
                dep_tree.nodes[node_id]['mask'] = 1
            else:
                dep_tree.nodes[node_id]['x'] = self.NO_ELEMENT
                dep_tree.nodes[node_id]['mask'] = 0

            if 'tag' in dep_tree.nodes[node_id]:
                node_tag = dep_tree.nodes[node_id]['tag']
                dep_tree.nodes[node_id]['tag_id'] = self.__get_tag_id__(node_tag)
            else:
                dep_tree.nodes[node_id]['tag_id'] = self.NO_ELEMENT

            sentiment_label = self.__get_sentiment__(phrase_subtree)

            assert len(list(dep_tree.successors(node_id))) <= 1
            dep_tree.nodes[node_id]['y'] = sentiment_label

            return phrase_subtree

        # find the root
        root_list = [x for x in dep_tree.nodes if dep_tree.out_degree(x) == 0]
        assert len(root_list) == 1
        _rec_visit_nx(root_list[0])

        assert dep_tree.nodes[root_list[0]]['y'] != -1

    def __parse_file__(self, f_name):
        with open(f_name, 'rb') as f:
            d = pickle.load(f)

        for nx_t in tqdm(d, desc='Loading trees'):
            g = self.preprocess_dep_tree(nx_t)
            self.tot_nodes += g.number_of_nodes()
            self.tot_leaves += len([i for i in range(g.number_of_nodes()) if g.in_degree(i) == 0])
            self.no_label += len([i for i in range(g.number_of_nodes()) if g.ndata['y'][i].item() == self.NO_ELEMENT])
            self.data.append(g)

    def __print_stats__(self):
        super().__print_stats__()
        self.logger.info('{} nodes loaded.'.format(self.tot_nodes))
        self.logger.info('{} leaves loaded.'.format(self.tot_leaves))
        self.logger.info('{} nodes without label.'.format(self.no_label))

    def __compute_retained_types__(self):
        min_occ = 10
        self.logger.info('Remove types that not occur more than {} times.'.format(min_occ))
        ret = ['but', 'neg']
        retained_types = [k for k, v in self.types_counting.items() if v >= min_occ and k != 'unk' and '_' not in k and k in ret]
        retained_types.insert(0, 'unk')

        return retained_types

    def get_loader(self, batch_size, device, shuffle=False):
        def batcher_dev(batch):
            batched_trees = dgl.batch(batch)
            batched_trees.to(device)
            return tuple([batched_trees]), batched_trees.ndata['y']

        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                          num_workers=0)


class SSTBinaryDataset(TreeDataset):

    def __init__(self, path_dir, file_name_list, logger, words_vocab=None, types_vocab=None):
        TreeDataset.__init__(self, path_dir, file_name_list, logger, words_vocab, types_vocab)
        self.tot_nodes = 0
        self.tot_leaves = 0
        self.no_label = 0
        self.num_classes = 5
        self.__load_trees__()

    # parse file and poupulate self.data
    def __parse_file__(self, f_name):

        with open(f_name, 'rb') as f:
            d = pickle.load(f)

        for nx_t in tqdm(d, desc='Loading trees'):
            self.__preprocess_tree__(nx_t)
            g = self.__to_dgl__(nx_t)
            self.tot_nodes += g.number_of_nodes()
            self.tot_leaves += len([i for i in range(g.number_of_nodes()) if g.in_degree(i) == 0])
            self.no_label += len(
                [i for i in range(g.number_of_nodes()) if g.ndata['y'][i].item() == self.NO_ELEMENT])
            self.data.append(g)

    def __preprocess_tree__(self, nx_t: nx.DiGraph):

        for u in nx_t.nodes:

            if 'word' in nx_t.nodes[u]:
                x = self.__get_word_id__(nx_t.nodes[u]['word'])
                mask = 1
            else:
                x = self.NO_ELEMENT
                mask = 0

            t = None
            if 'type' in nx_t.nodes[u]:
                type_list = list(set(nx_t.nodes[u]['type']))
                if len([x for x in type_list if 'neg' in x]):  # neg and conj:negcc
                    t = 'neg'
                elif len([x for x in type_list if ':' in x]) > 0:
                    t = [x for x in type_list if ':' in x][0].split(':')[1]
                else:
                    if len(type_list) > 0:
                        t = type_list[0]  # probabylu use UNK
            if t is not None:
                type_id = self.__get_type_id__(t)
            else:
                type_id = self.UNK

            nx_t.add_node(u, x=x, mask=mask, type_id=type_id)

    def __filter_types__(self):
        pass

    def __print_stats__(self):
        super().__print_stats__()
        self.logger.info('{} nodes loaded.'.format(self.tot_nodes))
        self.logger.info('{} leaves loaded.'.format(self.tot_leaves))
        self.logger.info('{} nodes without label.'.format(self.no_label))

    def get_loader(self, batch_size, device, shuffle=False):
        def batcher_dev(batch):
            batched_trees = dgl.batch(batch)
            batched_trees.to(device)
            return tuple([batched_trees]), batched_trees.ndata['y']

        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=batcher_dev, shuffle=shuffle,
                          num_workers=0)


#
#
# class SSTConstituencyDataset(SentenceTreeDataset):
#
#     def __init__(self, path_dir, file_name_list, name, words_vocab=None, tags_vocab=None):
#         SentenceTreeDataset.__init__(self, path_dir, file_name_list, name, words_vocab, tags_vocab)
#
#         self.__load_sentiment_mapping__()
#         self.__load_trees__()
#
#     def __build_dgl_tree__(self, root, i):
#         g = nx.DiGraph()
#
#         phrase_no_label = []
#
#         def _rec_build(node):
#             nonlocal phrase_no_label
#
#             if isinstance(node[0], str):
#                 # this is a leaf
#
#                 w = []
#                 for s in node:
#                     w = w + [s]
#                 w = '\xa0'.join(w)
#
#                 ch_id = g.number_of_nodes()
#                 word_id = self.__get_word_id__(w)
#                 sentiment_label = self.__get_sentiment__([w])
#                 tag_id = self.__get_tag_id__(node.label())
#                 g.add_node(ch_id, x=word_id, y=sentiment_label, mask=True, tag_id=tag_id)
#
#                 return [w], ch_id, None
#             else:
#                 # this is internal node
#                 phrase_subtree = []
#                 ch_id_list = []
#                 for i, child in enumerate(node):
#                     assert not isinstance(child, str)
#
#                     s, ch_id = _rec_build(child)
#                     ch_id_list.insert(i, ch_id)
#                     phrase_subtree += s
#
#                 node_id = g.number_of_nodes()
#                 sentiment_label = self.__get_sentiment__(phrase_subtree)
#                 tag_id = self.__get_tag_id__(node.label())
#                 g.add_node(node_id, x=SentenceTreeDataset.NO_ELEMENT, y=sentiment_label, mask=False, tag_id=tag_id)
#                 if sentiment_label == SentenceTreeDataset.NO_ELEMENT:
#                     phrase_no_label.append(phrase_subtree)
#
#                 # add edges
#                 assert len(ch_id_list) == len(node)
#                 self.max_out_degree = max(self.max_out_degree, len(node))
#                 for ch_id in ch_id_list:
#                     if ch_id is not None:
#                         g.add_edge(ch_id, node_id)
#
#                 return phrase_subtree, node_id
#
#         # skip first node
#         _rec_build(root[0])
#
#         assert -1 not in[g.nodes[i]['y'] for i in range(g.number_of_nodes()) if g.out_degree(i) == 0]
#
#         ret = dgl.DGLGraph()
#         ret.from_networkx(g, node_attrs=['x', 'y', 'mask', 'tag_id'])
#
#         return ret, len(phrase_no_label)
#

