import dgl


class BatcherFun:

    def __init__(self, only_root=False):
        self.only_root = only_root

    def __call__(self, tuple_data): #*args, **kwargs):
        tree_list = tuple_data
        batched_trees = dgl.batch(tree_list)
        if not self.only_root:
            out = batched_trees.ndata['y']
        else:
            root_ids = [i for i in range(batched_trees.number_of_nodes()) if batched_trees.out_degree(i) == 0]
            out = batched_trees.ndata['y'][root_ids]

        return [batched_trees], out
