from abc import abstractmethod

from torch.utils.data import Dataset

import os
import dgl
import networkx as nx


class TreeDataset(Dataset):

    NODES_ATTRIBUTE_LIST = ['x', 'y', 'mask', 'type_id']
    UNK = 0   # out-of-vocabulary element
    NO_ELEMENT = -1  # flag to indicate missing element

    def __init__(self, path_dir, file_name_list, logger, words_vocab=None, types_vocab=None):
        Dataset.__init__(self)
        self.data = []
        self.path_dir = path_dir
        self.file_name_list = file_name_list

        self.logger = logger

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


class DependencyTreeDataset(SentenceTreeDataset):

    def __init__(self, path_dir, file_name_list, name, do_augmentation='None', words_vocab=None, types_vocab=None,
                 tags_vocab=None):

        SentenceTreeDataset.__init__(self, path_dir, file_name_list, name, words_vocab, types_vocab, tags_vocab)

        self.do_augmentation = do_augmentation

    def preprocess_dep_tree(self, dep_graph: nx.DiGraph):

        try:
            rev_graph = dep_graph.reverse()
            # TODO: minimum_spanning_arborescence wrong answer on tree with id 7448
            # st_tree = nx.algorithms.minimum_spanning_arborescence(rev_graph, preserve_attrs=True)
            # st_tree = st_tree.reverse()

            dep_tree = nx.DiGraph()
            dep_tree.add_nodes_from([x for x in dep_graph.nodes(data=True)])
            succ_list = list(nx.algorithms.traversal.bfs_successors(rev_graph, 0))
            for pa, ch_list in succ_list:
                for ch in ch_list:
                    dep_tree.add_edge(ch, pa, **dep_graph.get_edge_data(ch, pa))

        except nx.exception.NetworkXException:
            raise ValueError

        # remove 0 node
        dep_tree.remove_node(0)

        assert nx.algorithms.tree.is_arborescence(dep_tree.reverse())

        nodes_to_remove = self.__get_id_nodes_to_remove__(dep_graph)

        if self.do_augmentation == 'branching' or self.do_augmentation == 'branching_sorted':

            if self.do_augmentation == 'branching_sorted':
                self.__transform_edge_features_adding_branching_nodes__(dep_tree, True)
            else:
                self.__transform_edge_features_adding_branching_nodes__(dep_tree, False)

            self.__compute_node_features__(dep_tree)

            rev_type_vocab = {v: k for k, v in self.types_vocab.items()}
            # remove nodes, removing also branching nodes if they have only on ch
            for u in nodes_to_remove:
                pa_u_list = list(dep_tree.successors(u))

                if dep_tree.nodes[u]['type_id'] != -1:
                    n_id = rev_type_vocab[dep_tree.nodes[u]['type_id']]
                    self.types_counting[n_id] -= 1

                dep_tree.remove_node(u)

                # TODO: check the following lines to remove the parent if it has one child
                assert len(pa_u_list) == 1  # u is never the root
                # try to remove also pa_u if it has only one child. (Useful to reduce tree depth)
                pa_u = pa_u_list[0]
                u_sibl = list(dep_tree.predecessors(pa_u))
                assert len(u_sibl) >=1
                if len(u_sibl) == 1 and ('neg' not in self.types_vocab or ('neg' in self.types_vocab and dep_tree.nodes[pa_u]['type_id'] != self.types_vocab['neg'])):
                    # we can remove also pa_u, if it is not the root and it is not neg
                    pa_pa_u_list = list(dep_tree.successors(pa_u))
                    if len(pa_pa_u_list) == 1:
                        pa_pa_u = pa_pa_u_list[0]
                        e_data = dep_tree.get_edge_data(pa_u, pa_pa_u)

                        if dep_tree.nodes[pa_u]['type_id'] != -1:
                            n_id = rev_type_vocab[dep_tree.nodes[pa_u]['type_id']]
                            self.types_counting[n_id] -= 1

                        dep_tree.add_edge(u_sibl[0], pa_pa_u, **e_data)
                    else:
                        # pa_u is the root! we must keep the y
                        # the new root is the onyl sibling of u
                        dep_tree.nodes[u_sibl[0]]['y'] = dep_tree.nodes[pa_u]['y']
                    dep_tree.remove_node(pa_u)
        else:
            # standard dependency tree
            # set type id to -1 to all nodes
            dep_tree.add_nodes_from([(x, dict(type_id=self.NO_ELEMENT)) for x in dep_tree.nodes])
            self.__compute_node_features__(dep_tree)

            # for u in nodes_to_remove:
            #     pred_list = list(dep_tree.predecessors(u))
            #     # we do not disconnect the tree
            #     if len(pred_list) == 0:
            #         dep_tree.remove_node(u)

        assert nx.algorithms.tree.is_arborescence(dep_tree.reverse())

        return self.__to_dgl__(dep_tree)

    def __compute_node_features__(self, dep_tree: nx.DiGraph):
        raise NotImplementedError

    def __transform_edge_features_adding_branching_nodes__(self, dep_tree: nx.DiGraph, sort_edges):

        all_nodes_id = list(nx.topological_sort(dep_tree))
        nid = max(all_nodes_id) + 1
        for u in all_nodes_id:
            # self.plot_tree(dep_tree)
            dep_tree.nodes[u]['type_id'] = self.__search_and_update_vocab__('types', 'leaf')

            pa_u = list(dep_tree.successors(u))[0] if list(dep_tree.successors(u)) else -1
            in_edge_list = list(dep_tree.in_edges(u, data=True))

            if sort_edges:

                def get_key(x):
                    t = x[2]['type']
                    if t == 'amod':
                        return -100
                    elif t == 'advmod':
                        return -80
                    elif t.startswith('nmod'):
                        return -50
                    elif t == 'nsubj':
                        return 15
                    elif t == 'ccomp':
                        return 17
                    elif (t.startswith('acl') or t.startswith('advcl')) and ':' in t:
                        return 20
                    elif t.startswith('conj'):
                        return 100
                    elif t == 'neg':
                        return 150
                    else:
                        return 0

                in_edge_list.sort(key=get_key)

            new_nodes = {}
            new_nodes_out = {}
            accumulator_node = u
            for v, _, edge_dict in in_edge_list:
                e_type = edge_dict['type']
                if ':' in e_type:
                    e_type = e_type.split(':')[1]
                if e_type not in new_nodes:
                    # create new node
                    dep_tree.add_node(nid, x=SentenceTreeDataset.NO_ELEMENT,
                                      y=SentenceTreeDataset.NO_ELEMENT,
                                      mask=False,
                                      tag_id=SentenceTreeDataset.NO_ELEMENT,
                                      type_id=self.__search_and_update_vocab__('types', e_type))
                    dep_tree.add_edge(accumulator_node, nid)
                    accumulator_node = nid
                    new_nodes[e_type] = nid
                    new_nodes_out[e_type] = 0
                    nid += 1

                new_nodes_out[e_type] += 1
                mid_node = new_nodes[e_type]
                dep_tree.add_edge(v, mid_node)
                dep_tree.remove_edge(v, u)

            if accumulator_node != u and pa_u != -1:
                pa_u_edge_data = dep_tree.get_edge_data(u, pa_u)
                dep_tree.add_edge(accumulator_node, pa_u, **pa_u_edge_data)
                dep_tree.remove_edge(u, pa_u)
                dep_tree.add_edge(accumulator_node, pa_u)

    def __get_id_nodes_to_remove__(self, dep_graph: nx.DiGraph):

        nodes_to_remove = set()

        # TODO: there are punct that are not leaves
        types_of_edge_to_remove = ['det', 'punct', 'neg'] # 'mark', 'case', 'cc', 'nwe', 'ref', 'cop', 'aux']

        for u, v, d in dep_graph.edges(data=True):
            type = d['type']
            if type in types_of_edge_to_remove:
                #pred_u = nx.algorithms.dag.ancestors(dep_graph, u)
                nodes_to_remove.add(u)

        # for u, v, d in dep_graph.edges(data=True):
        #     type = d['type']
        #     if ':' in type:
        #         splitted_type = type.split(':')
        #         to_remove = -1
        #         if splitted_type[0] in ['nmod', 'advcl', 'acl']:
        #             # there is a child of u which contains the word
        #             to_remove = u
        #         elif splitted_type[0] in ['conj']:
        #             # there is a child of v which contains the word
        #             to_remove = v
        #
        #         if to_remove != -1:
        #             for x in dep_graph.predecessors(to_remove):
        #                 if dep_graph.nodes[x]['word'].lower() == splitted_type[1]:
        #                     nodes_to_remove.add(x)
        #                     nodes_to_remove.update(nx.algorithms.dag.ancestors(dep_graph, x))

        return nodes_to_remove
