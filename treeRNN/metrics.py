from abc import ABC, abstractmethod
import torch as th
import copy


class BaseMetric:

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
    def is_better_than(self, other_metric):
        raise NotImplementedError('users must define update_metrics to use this base class')


class TreeMetric(BaseMetric):

    @abstractmethod
    def update_metric(self, out, gold_label, graph):
        raise NotImplementedError('users must define update_metrics to use this base class')


class ValueMetric(BaseMetric):

    @abstractmethod
    def update_metric(self, out, gold_label):
        raise NotImplementedError('users must define update_metrics to use this base class')


class Accuracy(ValueMetric):

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

    def is_better_than(self, other_metric):
        return self.final_value > other_metric.final_value


class RootAccuracy(TreeMetric):

    def initialise_metric(self):
        self.n_nodes = 0
        self.n_correct = 0

    def update_metric(self, out, gold_label, graph):
        root_ids = [i for i in range(graph.number_of_nodes()) if graph.out_degree(i) == 0]
        pred = th.argmax(out, 1)
        self.n_correct += th.sum(th.eq(pred[root_ids], gold_label[root_ids])).item()
        self.n_nodes += len(root_ids)

    def finalise_metric(self):
        self.final_value = self.n_correct / self.n_nodes

    def __str__(self):
        return "Root Accuracy: {:4f}".format(self.final_value)

    def is_better_than(self, other_metric):
        return self.final_value > other_metric.final_value


class LeavesAccuracy(TreeMetric):

    def initialise_metric(self):
        self.n_nodes = 0
        self.n_correct = 0

    def update_metric(self, out, gold_label, graph):
        leaves_ids = [i for i in range(graph.number_of_nodes()) if graph.in_degree(i) == 0]
        pred = th.argmax(out, 1)
        self.n_correct += th.sum(th.eq(pred[leaves_ids], gold_label[leaves_ids])).item()
        self.n_nodes += len(leaves_ids)

    def finalise_metric(self):
        self.final_value = self.n_correct / self.n_nodes

    def __str__(self):
        return "Leaves Accuracy: {:4f}".format(self.final_value)

    def is_better_than(self, other_metric):
        return self.final_value > other_metric.final_value


class MSE(ValueMetric):

    def initialise_metric(self):
        self.val = 0
        self.n_val = 0

    def update_metric(self, out, gold_label):
        self.val += th.sum((out-gold_label).pow(2)).item()
        self.n_val += len(gold_label)

    def finalise_metric(self):
        self.final_value = self.val / self.n_val

    def __str__(self):
        return "MSE: {:4f}".format(self.final_value)

    def is_better_than(self, other_metric):
        return self.final_value < other_metric.final_value


class Pearson(ValueMetric):

    def initialise_metric(self):
        self.x = None
        self.y = None

    def update_metric(self, out, gold_label):
        x = copy.deepcopy(out)
        y = copy.deepcopy(gold_label)

        if self.x is None:
            self.x = x
        else:
            self.x = th.cat((self.x, x), dim=0)

        if self.y is None:
            self.y = y
        else:
            self.y = th.cat((self.y, y), dim=0)

    def finalise_metric(self):

        vx = self.x - th.mean(self.x)
        vy = self.y - th.mean(self.y)

        cost = th.sum(vx * vy) / (th.sqrt(th.sum(vx ** 2)) * th.sqrt(th.sum(vy ** 2)))
        self.final_value = cost.item()

    def __str__(self):
        return "Pearson: {:4f}".format(self.final_value)

    def is_better_than(self, other_metric):
        return self.final_value > other_metric.final_value


class MaskedAccuracy(Accuracy):
    NO_ELEMENT = -1

    def update_metric(self, out, gold_label):
        pred = th.argmax(out, 1)
        idxs = (gold_label != MaskedAccuracy.NO_ELEMENT)
        self.n_correct += th.sum(th.eq(gold_label[idxs], pred[idxs])).item()
        self.n_nodes += th.sum(idxs).item()
