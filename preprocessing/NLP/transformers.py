from abc import abstractmethod
import networkx as nx
from utils.serialization import from_json_file


class BaseTransformer:

    # indicates if the trandormation creates node types information
    CREATE_TYPES = None

    @abstractmethod
    def transform(self, *args):
        raise NotImplementedError('This method must be implemented in a sub class')


class DepTreeTransformer(BaseTransformer):

    CREATE_TYPES = False

    def __init__(self):
        super(DepTreeTransformer, self).__init__()

    def transform(self, t):

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

        assert nx.algorithms.tree.is_arborescence(dep_tree.reverse())

        return dep_tree


class ConstTreeTransformer(BaseTransformer):

    CREATE_TYPES = True

    def __init__(self, pos_tag_clusters_file=None, max_out_degree=None):
        super(ConstTreeTransformer, self).__init__()

        if pos_tag_clusters_file is not None:
            pos_tag_clusters = from_json_file(pos_tag_clusters_file)
            self.tag2cluster = {}
            for k, v in pos_tag_clusters.items():
                self.tag2cluster.update({vv: k for vv in v})
        else:
            self.tag2cluster = None

        self.max_out_degree = max_out_degree

    def transform(self, t):
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

        # remove composed tag obtained from binarisation and copy tag to type attribute
        for u, d in new_t.nodes(data=True):
            tag = d['tag']
            if '|' in tag:
                tag = tag.split('|')[0]

            if self.tag2cluster is not None:
                tag = self.tag2cluster[tag] if tag in self.tag2cluster else 'X'

            d['type'] = tag


        # if max_out_degree is set, remove all child nodes with a position greater than that
        if self.max_out_degree is not None:
            idx_to_remove = set()

            for u in new_t.nodes:
                all_ch = list(new_t.predecessors(u))
                all_ch.sort()
                for ch in all_ch[self.max_out_degree:]:
                    idx_to_remove = idx_to_remove.union(set(nx.ancestors(new_t, ch)))
                    idx_to_remove.add(ch)

            sss = new_t.copy(as_view=False)
            new_t.remove_nodes_from(idx_to_remove)

        assert nx.algorithms.tree.is_arborescence(new_t.reverse())
        assert len([u for u, d in new_t.nodes(data=True) if 'tag' not in d]) == 0
        return new_t


class CombinedTreeTransformer(BaseTransformer):

    CREATE_TYPES = True

    def __init__(self, split_on_semicolon=True):
        super(CombinedTreeTransformer, self).__init__()
        self.const_tree_transformer = ConstTreeTransformer()
        self.split_on_semicolon = split_on_semicolon

    def transform(self, bin_t, dep_t):
        bin_t = self.const_tree_transformer.transform(bin_t)
        for u, d in bin_t.nodes(data=True):
            del d['type']

        dep_t = self.__remove_dep_edges__(dep_t)

        const_leaves = sorted([(i, d) for i, d in bin_t.nodes(data=True) if 'token_id' in d], key=lambda x: x[1]['token_id'])
        assert (bin_t.number_of_nodes() - len(const_leaves)) == dep_t.number_of_edges()-1

        new_bin_t = bin_t.copy(as_view=False)

        rev_const_t = new_bin_t.reverse(copy=False)

        u_types = {}

        for u, v, d in dep_t.edges(data=True):
            edge_type = d['type']
            if v == 0:
                # skip edges to the root
                continue

            id_token_u = dep_t.nodes[u]['token_id']
            id_token_v = dep_t.nodes[v]['token_id']

            u_in_const = const_leaves[id_token_u - 1][0]
            v_in_const = const_leaves[id_token_v - 1][0]

            assert id_token_u == bin_t.nodes[u_in_const]['token_id']
            assert id_token_v == bin_t.nodes[v_in_const]['token_id']

            assert dep_t.nodes[u]['word'].lower() == bin_t.nodes[u_in_const]['word'].lower()
            assert dep_t.nodes[v]['word'].lower() == bin_t.nodes[v_in_const]['word'].lower()

            if dep_t.in_degree(u) == 0:
                # u is a leaf
                lca = list(bin_t.successors(u_in_const))[0]
            else:
                lca = nx.algorithms.lowest_common_ancestor(rev_const_t, u_in_const, v_in_const)
            assert lca is not None

            if lca not in u_types:
                u_types[lca] = {}

            u_types[lca][(id_token_u, id_token_v)] = edge_type

            if 'type' in new_bin_t.nodes[lca]:
                if not isinstance(new_bin_t.nodes[lca]['type'], list):
                    new_bin_t.nodes[lca]['type'] = [new_bin_t.nodes[lca]['type']]
                new_bin_t.nodes[lca]['type'] += [edge_type]
            else:
                new_bin_t.add_node(lca, type=edge_type)

        # compute token id set for each node
        u_token_list = {}
        for u in nx.algorithms.topological_sort(new_bin_t):
            if 'token_id' in new_bin_t.nodes[u]:
                u_token_list[u] = [new_bin_t.nodes[u]['token_id']]
            else:
                u_token_list[u] = []
                for ch in new_bin_t.predecessors(u):
                    u_token_list[u] += u_token_list[ch]

        for u, d_types in u_types.items():
            if len(d_types) == 1:
                # nodes with only one type are ok
                continue
            del new_bin_t.nodes[u]['type']
            # we sort keys in order to follow the text order (keys are toke id)
            for k in sorted(d_types.keys()):
                x = u
                # search in the path between u and k[0], a node with no type
                while 'type' in new_bin_t.nodes[x]:
                    p_list = list(new_bin_t.predecessors(x))
                    if k[0] in u_token_list[p_list[0]]:
                        x = p_list[0]
                    elif k[0] in u_token_list[p_list[1]]:
                        x = p_list[1]
                    else:
                        raise ValueError
                new_bin_t.nodes[x]['type'] = d_types[k]

        if self.split_on_semicolon:
            # if a type has :, split it and take the last part
            for u, d in new_bin_t.nodes(data=True):
                if 'type' in d and ':' in d['type']:
                    d['type'] = d['type'].split(':')[1]

        return new_bin_t

    @staticmethod
    def __remove_dep_edges__(dep_t: nx.DiGraph):
        new_dep_t = dep_t.copy(as_view=False)
        all_edges = list(new_dep_t.edges(data=True))

        for u, v, d in all_edges:
            # remove nsubj:xsubj edges
            if d['type'] == 'nsubj:xsubj':
                new_dep_t.remove_edge(u, v)
            # remove self-edges
            if u == v:
                new_dep_t.remove_edge(u,v)

        # remove propagation
        for u in new_dep_t.nodes:
            all_succ = list(new_dep_t.successors(u))
            if len(all_succ) > 1:
                #all_succ = sorted(all_succ, key=lambda x: nodes_order[x])
                for v in all_succ:
                    ev = new_dep_t.edges[u,v]['type']
                    for vv in all_succ:
                        if new_dep_t.has_edge(vv, v) and ev == new_dep_t.edges[vv,v]['type']:
                            # (u,v) is a propagation
                            new_dep_t.remove_edge(u, v)

        nodes_order = {0: 0}
        for u, ch_list in nx.algorithms.bfs_successors(new_dep_t.reverse(copy=False), 0):
            for ch in ch_list:
                assert ch not in nodes_order
                nodes_order[ch] = len(nodes_order)

        assert len(nodes_order) == new_dep_t.number_of_nodes()

        # chek if there are nodes with more than on parent
        for u in new_dep_t.nodes:
            succ_list = list(new_dep_t.successors(u))
            if len(succ_list) > 1:
                succ_list = sorted(succ_list, key=lambda x: nodes_order[x])
                for v in succ_list[1:]:
                    new_dep_t.remove_edge(u, v)

        return new_dep_t


class TreeNetTransformer(ConstTreeTransformer):

    CREATE_TYPES = False

    def __init__(self):
        super(TreeNetTransformer, self).__init__(None)

    def transform(self, t:nx.DiGraph):
        # make a copy
        t = super(TreeNetTransformer, self).transform(t)

        t = nx.reverse(t)
        new_t = nx.DiGraph()
        new_t.add_nodes_from(t.nodes(data=True))
        f_id = max(list(t.nodes)) +1
        new_t.add_node(f_id)
        new_id = f_id+1
        for u in nx.topological_sort(t):
            prev = u
            child_list = sorted(list(t.successors(u)), reverse=True)
            for v in child_list:
                if prev == u:
                    pos = 1
                else:
                    pos = 0
                if t.out_degree(v) == 0:
                    # v is leaf
                    pa_v = new_id
                    new_id += 1

                    new_t.add_node(pa_v)
                    new_t.add_edge(v, pa_v, pos=1)
                    new_t.add_edge(pa_v, prev, pos=pos)
                    prev = pa_v
                else:
                    # v is internal
                    new_t.add_edge(v, prev, pos=pos)
                    prev = v

        for u in new_t.nodes:
            if new_t.in_degree(u) == 1:
                ee = new_t.edges[list(new_t.in_edges(u))[0]]
                new_t.add_edge(f_id, u, pos=(ee['pos']+1)%2)
            assert new_t.in_degree(u) == 2 or new_t.in_degree(u) == 0

        return new_t

