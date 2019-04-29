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
    def update_metric(self, out, batch):
        raise NotImplementedError('users must define update_metrics to use this base class')

    @abstractmethod
    def finalise_metric(self):
        raise NotImplementedError('users must define finalise_metric to use this base class')

    @abstractmethod
    def is_better_than(self, value):
        raise NotImplementedError('users must define update_metrics to use this base class')


class LabelAccuracy(BaseMetric):

    def __init__(self):
        BaseMetric.__init__(self)
        self.n_nodes = 0
        self.n_correct = 0

    def initialise_metric(self):
        self.n_nodes = 0
        self.n_correct = 0

    def update_metric(self, out, batch):
        pred = th.argmax(out, 1)
        self.n_correct += th.sum(th.eq(batch.label, pred)).item()
        self.n_nodes += len(batch.label)

    def finalise_metric(self):
        self.final_value = self.n_correct / self.n_nodes

    def __str__(self):
        return "Nodes Accuracy: {:4f}".format(self.final_value)

    def is_better_than(self, value):
        return self.final_value > value



class RootAccuracy(BaseMetric):

    def __init__(self):
        BaseMetric.__init__(self)
        self.n_nodes = 0
        self.n_correct = 0

    def initialise_metric(self):
        self.n_nodes = 0
        self.n_correct = 0

    def update_metric(self, out, batch):
        root_ids = [i for i in range(batch.graph.number_of_nodes()) if batch.graph.out_degree(i) == 0]
        # leaves_ids = [i for i in range(batch.graph.number_of_nodes()) if batch.graph.in_degree(i) == 0]
        pred = th.argmax(out, 1)
        root_pred = pred.cpu().data.numpy()[root_ids]
        root_labels = batch.label.cpu().data.numpy()[root_ids]
        self.n_correct += np.sum(root_labels == root_pred)
        self.n_nodes += len(root_pred)

    def finalise_metric(self):
        self.final_value = self.n_correct / self.n_nodes

    def __str__(self):
        return "Root Accuracy: {:4f}".format(self.final_value)

    def is_better_than(self, value):
        return self.final_value > value


class LeavesAccuracy(BaseMetric):

    def __init__(self):
        BaseMetric.__init__(self)
        self.n_nodes = 0
        self.n_correct = 0

    def initialise_metric(self):
        self.n_nodes = 0
        self.n_correct = 0

    def update_metric(self, out, batch):
        leaves_ids = [i for i in range(batch.graph.number_of_nodes()) if batch.graph.in_degree(i) == 0]
        pred = th.argmax(out, 1)
        leaves_pred = pred.cpu().data.numpy()[leaves_ids]
        leaves_labels = batch.label.cpu().data.numpy()[leaves_ids]
        self.n_correct += np.sum(leaves_labels == leaves_pred)
        self.n_nodes += len(leaves_pred)

    def finalise_metric(self):
        self.final_value = self.n_correct / self.n_nodes

    def __str__(self):
        return "Leaves Accuracy: {:4f}".format(self.final_value)

    def is_better_than(self, value):
        return self.final_value > value