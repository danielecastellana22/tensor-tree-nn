from experiments.base import ProbExperiment, NeuralExperiment
import dgl


def batcher_dev(tuple_data):
    tree_list = tuple_data
    batched_trees = dgl.batch(tree_list)
    out = batched_trees.ndata['y']

    return [batched_trees], out


class ProbToyExperiment(ProbExperiment):

    def __get_batcher_function__(self):
        return batcher_dev


class NeuralToyExperiment(NeuralExperiment):

    def __get_batcher_function__(self):
        return batcher_dev