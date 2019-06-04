from abc import ABC, abstractmethod
import torch as th
import numpy as np


class BaseMetric(ABC):

    def __init__(self):
        self.final_value = None
        self.initialise_metric()

    def get_value(self):
        return self.final_value

    @abstractmethod
    def initialise_metric(self):
        raise NotImplementedError('users must define update_metrics to use this base class')

    @abstractmethod
    def finalise_metric(self):
        raise NotImplementedError('users must define finalise_metric to use this base class')

    @abstractmethod
    def is_better_than(self, value):
        raise NotImplementedError('users must define update_metrics to use this base class')


class TreeMetric(BaseMetric):

    def __init__(self):
        BaseMetric.__init__()


    @abstractmethod
    def update_metric(self, out, gold_label, graph):
        raise NotImplementedError('users must define update_metrics to use this base class')


class ValueMetric(BaseMetric):
    def __init__(self):
        BaseMetric.__init__()

    @abstractmethod
    def update_metric(self, out, gold_label):
        raise NotImplementedError('users must define update_metrics to use this base class')


class Accuracy(ValueMetric):

    def __init__(self):
        BaseMetric.__init__(self)
        self.n_nodes = 0
        self.n_correct = 0

    def initialise_metric(self):
        self.n_nodes = 0
        self.n_correct = 0

    def update_metric(self, out, gold_label):
        pred = th.argmax(out, 1)
        self.n_correct += th.sum(th.eq(gold_label, pred)).item()
        self.n_nodes += len(gold_label)

    def finalise_metric(self):
        self.final_value = self.n_correct / self.n_nodes

    def __str__(self):
        return "Accuracy: {:4f}".format(self.final_value)

    def is_better_than(self, value):
        return self.final_value > value


class RootAccuracy(TreeMetric):

    def __init__(self):
        BaseMetric.__init__(self)
        self.n_nodes = 0
        self.n_correct = 0

    def initialise_metric(self):
        self.n_nodes = 0
        self.n_correct = 0

    def update_metric(self, out, gold_label, graph):
        root_ids = [i for i in range(graph.number_of_nodes()) if graph.out_degree(i) == 0]
        # leaves_ids = [i for i in range(batch.graph.number_of_nodes()) if batch.graph.in_degree(i) == 0]
        pred = th.argmax(out, 1)
        root_pred = pred.cpu().data.numpy()[root_ids]
        root_labels = gold_label.cpu().data.numpy()[root_ids]
        self.n_correct += np.sum(root_labels == root_pred)
        self.n_nodes += len(root_pred)

    def finalise_metric(self):
        self.final_value = self.n_correct / self.n_nodes

    def __str__(self):
        return "Root Accuracy: {:4f}".format(self.final_value)

    def is_better_than(self, value):
        return self.final_value > value


class LeavesAccuracy(TreeMetric):

    def __init__(self):
        BaseMetric.__init__(self)
        self.n_nodes = 0
        self.n_correct = 0

    def initialise_metric(self):
        self.n_nodes = 0
        self.n_correct = 0

    def update_metric(self, out, gold_label, graph):
        leaves_ids = [i for i in range(graph.number_of_nodes()) if graph.in_degree(i) == 0]
        pred = th.argmax(out, 1)
        leaves_pred = pred.cpu().data.numpy()[leaves_ids]
        leaves_labels = gold_label.cpu().data.numpy()[leaves_ids]
        self.n_correct += np.sum(leaves_labels == leaves_pred)
        self.n_nodes += len(leaves_pred)

    def finalise_metric(self):
        self.final_value = self.n_correct / self.n_nodes

    def __str__(self):
        return "Leaves Accuracy: {:4f}".format(self.final_value)

    def is_better_than(self, value):
        return self.final_value > value


class MSE(ValueMetric):

    def __init__(self):
        BaseMetric.__init__(self)
        self.val = 0
        self.n_val = 0

    def initialise_metric(self):
        self.val = 0
        self.n_val = 0

    def update_metric(self, out, gold_label):
        self.val += th.sum((out-gold_label).pow(2))
        self.n_val += len(gold_label)

    def finalise_metric(self):
        self.final_value = self.val / self.n_val

    def __str__(self):
        return "MSE: {:4f}".format(self.final_value)

    def is_better_than(self, value):
        return self.final_value < value
