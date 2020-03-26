from abc import abstractmethod
import torch as th
import copy
from preprocessing.utils import ConstValues


class BaseMetric:
    # TODO: get_name to avoide __name__ from outside
    def __init__(self):
        self.final_value = None

    def get_value(self):
        return self.final_value

    def is_better_than(self, other_metric):
        if self.HIGHER_BETTER:
            return self.final_value > other_metric.final_value
        else:
            return self.final_value < other_metric.final_value

    def __str__(self):
        return "{}: {:4f}".format(type(self).__name__, self.final_value)

    @abstractmethod
    def finalise_metric(self):
        raise NotImplementedError('users must define finalise_metric to use this base class')


class TreeMetricUpdate:

    @abstractmethod
    def update_metric(self, out, gold_label, graph):
        raise NotImplementedError('users must define update_metrics to use this base class')


class ValueMetricUpdate:

    @abstractmethod
    def update_metric(self, out, gold_label):
        raise NotImplementedError('users must define update_metrics to use this base class')


class BaseAccuracy(BaseMetric):
    HIGHER_BETTER = True

    def __init__(self):
        super(BaseMetric, self).__init__()
        self.n_nodes = 0
        self.n_correct = 0

    def finalise_metric(self):
        self.final_value = self.n_correct / self.n_nodes


class Accuracy(BaseAccuracy, ValueMetricUpdate):

    def __init__(self):
        super(Accuracy, self).__init__()

    def update_metric(self, out, gold_label: th.Tensor):
        pred = th.argmax(out, 1)
        idxs = (gold_label != ConstValues.NO_ELEMENT)
        self.n_correct += th.sum(th.eq(gold_label[idxs], pred[idxs])).item()
        self.n_nodes += th.sum(idxs).item()


class RootAccuracy(BaseAccuracy, TreeMetricUpdate):

    def __init__(self):
        super(RootAccuracy, self).__init__()

    def update_metric(self, out, gold_label, graph):
        root_ids = [i for i in range(graph.number_of_nodes()) if graph.out_degree(i) == 0]
        pred = th.argmax(out, 1)
        self.n_correct += th.sum(th.eq(pred[root_ids], gold_label[root_ids])).item()
        self.n_nodes += len(root_ids)


class LeavesAccuracy(BaseAccuracy, TreeMetricUpdate):

    def __init__(self):
        super(LeavesAccuracy, self).__init__()

    def update_metric(self, out, gold_label, graph):
        leaves_ids = [i for i in range(graph.number_of_nodes()) if graph.in_degree(i) == 0]
        pred = th.argmax(out, 1)
        self.n_correct += th.sum(th.eq(pred[leaves_ids], gold_label[leaves_ids])).item()
        self.n_nodes += len(leaves_ids)


class MSE(BaseMetric, ValueMetricUpdate):

    HIGHER_BETTER = False

    def __init__(self):
        super(MSE, self).__init__()
        self.val = 0
        self.n_val = 0

    def update_metric(self, out, gold_label):
        self.val += th.sum((out-gold_label).pow(2)).item()
        self.n_val += len(gold_label)

    def finalise_metric(self):
        self.final_value = self.val / self.n_val


class Pearson(BaseMetric, ValueMetricUpdate):

    HIGHER_BETTER = True

    def __init__(self):
        super(Pearson, self).__init__()
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


class RootChildrenAccuracy(BaseAccuracy, TreeMetricUpdate):

    def initialise_metric(self):
        super(RootChildrenAccuracy, self).__init__()

    def update_metric(self, out, gold_label, graph):
        root_ids = [i for i in range(graph.number_of_nodes()) if graph.out_degree(i) == 0]
        root_ch_id = [i for i in range(graph.number_of_nodes()) if i not in root_ids and graph.successors(i).item() in root_ids]

        pred = th.argmax(out, 1)
        self.n_correct += th.sum(th.eq(pred[root_ch_id], gold_label[root_ch_id])).item()
        self.n_nodes += len(root_ch_id)
