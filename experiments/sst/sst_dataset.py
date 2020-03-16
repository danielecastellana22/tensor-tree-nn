import networkx as nx
import dgl
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import namedtuple
from treeRNN.dataset import DependencyTreeDataset, TreeDataset
import pickle
import os


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


if __name__ == '__main__':
    # create the sst dataset

    data_dir = 'data/sst/bin_tree'
    output_dir = 'data/sst/dgl_bin_tree'

    # create word_vocab
    words_vocab = {'unk': 0}
    with open(os.path.join(data_dir, 'vocab.txt')) as f:
        for l in f.readlines():
            l = l.strip()
            words_vocab[l] = len(words_vocab)

    # create type
    for f in ['train.pkl', 'test.pkl']



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

