from abc import abstractmethod
import networkx as nx


class BaseTransfomer:

    @abstractmethod
    def transform(self, t: nx.DiGraph, *args):
        raise NotImplementedError('This method must be implemented in a sub class')


class DepTreeTransformer(BaseTransfomer):

    def __init__(self, do_augmentation='None'):
        self.do_augmentation = do_augmentation

    def transform(self, t, *args):

        try:
            rev_graph = t.reverse()
            # TODO: minimum_spanning_arborescence wrong answer on tree with id 7448
            # st_tree = nx.algorithms.minimum_spanning_arborescence(rev_graph, preserve_attrs=True)
            # st_tree = st_tree.reverse()

            dep_tree = nx.DiGraph()
            dep_tree.add_nodes_from([x for x in t.nodes(data=True)])
            succ_list = list(nx.algorithms.traversal.bfs_successors(rev_graph, 0))
            for pa, ch_list in succ_list:
                for ch in ch_list:
                    dep_tree.add_edge(ch, pa, **t.get_edge_data(ch, pa))

        except nx.exception.NetworkXException:
            raise ValueError

        # remove 0 node
        dep_tree.remove_node(0)

        assert nx.algorithms.tree.is_arborescence(dep_tree.reverse())

        nodes_to_remove = self.__get_id_nodes_to_remove__(t)

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
            pass
            # set type id to -1 to all nodes
            #dep_tree.add_nodes_from([(x, dict(type_id=self.NO_ELEMENT)) for x in dep_tree.nodes])
            #self.__compute_node_features__(dep_tree)

            # for u in nodes_to_remove:
            #     pred_list = list(dep_tree.predecessors(u))
            #     # we do not disconnect the tree
            #     if len(pred_list) == 0:
            #         dep_tree.remove_node(u)

        assert nx.algorithms.tree.is_arborescence(dep_tree.reverse())

        return dep_tree

    def __transform_edge_features_adding_branching_nodes__(self, dep_tree: nx.DiGraph, sort_edges):

        all_nodes_id = list(nx.topological_sort(dep_tree))
        nid = max(all_nodes_id) + 1
        for u in all_nodes_id:
            # self.plot_tree(dep_tree)
            dep_tree.nodes[u]['type_id'] = 'leaf'

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
                    dep_tree.add_node(nid, type_id=e_type)
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


class ConstTreeTransformer(BaseTransfomer):

    def transform(self, t, *args):
        # make a copy
        new_t = t.copy(as_view=False)

        # remove root node
        root_id = [u for u in new_t.nodes if new_t.out_degree(u) == 0]
        assert len(root_id) == 1
        root_id = root_id[0]
        assert new_t.in_degree(root_id) == 1
        new_t.remove_node(root_id)

        # collapse unary nodes
        unary_nodes = [u for u, d in new_t.in_degree if d==1]
        while len(unary_nodes) > 0:

            for pa_u in unary_nodes:
                u = list(new_t.predecessors(pa_u))[0]

                if 'tag' not in new_t.nodes[u]:
                    # copy the tag in the children leaf node
                    new_t.add_node(u, tag=new_t.nodes[pa_u]['tag'])

                # remove pa_u

                if len(list(new_t.successors(pa_u))) == 1:
                    pa_pa_u = list(new_t.successors(pa_u))[0]
                    new_t.add_edge(u, pa_pa_u)
                new_t.remove_node(pa_u)

            unary_nodes = [u for u, d in new_t.in_degree if d == 1]

        assert nx.algorithms.tree.is_arborescence(new_t.reverse())
        return new_t


class SuperTreeTransformer(BaseTransfomer):

    def __init__(self, type):
        super(SuperTreeTransformer, self).__init__()
        self.type = type

    def transform(self, t: nx.DiGraph, *args):
        if self.type == 'const_tag':
            return self.__const_tag_transform__(t, args[0])
        else:
            return self.__dep_rel_transform__(t, args[0])

    def __const_tag_transform__(self, bin_t: nx.DiGraph, const_t: nx.DiGraph):
        phrase2tag = self.__get_prhase2tag_dict(const_t)

        def _rec_assign_(node_id):
            all_ch = list(bin_t.predecessors(node_id))
            phrase_subtree = []

            if len(all_ch) == 0:
                # Leaf
                node_word = bin_t.nodes[node_id]['word'].lower()
                phrase_subtree += [node_word]
            else:
                for ch_id in all_ch:
                    s = _rec_assign_(ch_id)
                    phrase_subtree += s

            key = tuple(sorted(list(set(phrase_subtree))))
            if key in phrase2tag:
                bin_t.add_node(node_id, tag=phrase2tag[key])

            return phrase_subtree

        root_list = [x for x in bin_t.nodes if bin_t.out_degree(x) == 0]
        assert len(root_list) == 1
        _rec_assign_(root_list[0])

        from utils.visualisation import plot_netwrokx_tree
        import  matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 2, figsize=(20, 10))
        plot_netwrokx_tree(bin_t, node_attr=['word', 'tag'], ax=ax[0])
        plot_netwrokx_tree(const_t, node_attr=['word', 'tag'], ax=ax[1])
        plt.show()
        asddd=3

    def __dep_rel_transform__(self, bin_t: nx.DiGraph, dep_t: nx.DiGraph):
        sort_fun = lambda x: x[1]['token_id']
        const_leaves = sorted([(i, d) for i, d in const_t.nodes(data=True) if 'token_id' in d], key=sort_fun)
        new_const_t = copy.deepcopy(const_t)

        rev_const_t = new_const_t.reverse(copy=False)
        n_wrong = 0
        for u, v, d in dep_t.edges(data=True):
            t = d['type']
            if v == 0:
                continue
            try:
                id_token_u = dep_t.nodes[u]['token_id']
                id_token_v = dep_t.nodes[v]['token_id']

                u_in_const = const_leaves[id_token_u - 1][0]
                v_in_const = const_leaves[id_token_v - 1][0]

                # get the tag
                new_const_t.add_node(u_in_const, tag=dep_t.nodes[u]['tag'])

                assert id_token_u == const_t.nodes[u_in_const]['token_id']
                assert id_token_v == const_t.nodes[v_in_const]['token_id']

                assert dep_t.nodes[u]['word'] == const_t.nodes[u_in_const]['word']
                assert dep_t.nodes[v]['word'] == const_t.nodes[v_in_const]['word']
                lca = nx.algorithms.lowest_common_ancestor(rev_const_t, u_in_const, v_in_const)
                assert lca is not None

                if 'type' in new_const_t.nodes[lca]:
                    new_const_t.nodes[lca]['type'].append(t)
                else:
                    new_const_t.add_node(lca, type=[t])
            except:
                n_wrong += 1

        return new_const_t, n_wrong


    @staticmethod
    def __get_prhase2tag_dict(const_t: nx.DiGraph):
        out_dict = {}

        def _rec_visit_nx(node_id):
            nonlocal out_dict
            all_ch = list(const_t.predecessors(node_id))
            phrase_subtree = []

            if len(all_ch) == 0:
                # Leaf
                node_word = const_t.nodes[node_id]['word'].lower()
                phrase_subtree += [node_word]
            else:
                for ch_id in all_ch:
                    s = _rec_visit_nx(ch_id)
                    phrase_subtree += s

                tag = const_t.nodes[node_id]['tag']
                key = tuple(sorted(list(set(phrase_subtree))))
                if key not in out_dict:
                    out_dict[key] = tag

            return phrase_subtree

        # find the root
        root_list = [x for x in const_t.nodes if const_t.out_degree(x) == 0]
        assert len(root_list) == 1
        _rec_visit_nx(root_list[0])
        return out_dict

