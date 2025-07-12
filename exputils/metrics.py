from abc import abstractmethod, ABC
import torch as th
import copy

from exputils.datasets import ConstValues


class BaseMetric:

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

    def __add__(self, other):
        t = type(self)
        # check same type
        if type(other) is t:
            o = t()
            o.final_value = self.final_value + other.final_value
            return o
        else:
            raise TypeError('The metrics must have the same types!')

    def __truediv__(self, num):
        t = type(self)
        o = t()
        o.final_value = self.final_value / num
        return o

    @classmethod
    def get_name(cls):
        return cls.__name__

    @abstractmethod
    def update_metric(self, y_pred, y_true, in_data):
        raise NotImplementedError('users must define update_metrics to use this base class')

    @abstractmethod
    def finalise_metric(self):
        raise NotImplementedError('users must define finalise_metric to use this base class')


class Accuracy(BaseMetric):
    HIGHER_BETTER = True

    def __init__(self):
        super(BaseMetric, self).__init__()
        self.n_val = 0
        self.n_correct = 0

    def update_metric(self, y_pred, y_true: th.Tensor, in_data):
        pred = th.argmax(y_pred, 1)
        mask = (y_true != ConstValues.NO_ELEMENT)
        self.n_correct += th.sum(th.eq(y_true[mask], pred[mask])).item()
        self.n_val += th.sum(mask).item()

    def finalise_metric(self):
        if self.n_val != 0:
            self.final_value = self.n_correct / self.n_val
        else:
            self.final_value = 0


class RootAccuracy(Accuracy):

    def update_metric(self, y_pred, y_true, in_data):
        root_ids = [i for i in range(in_data.number_of_nodes()) if in_data.out_degree(i) == 0]
        super(RootAccuracy, self).update_metric(y_pred[root_ids], y_true[root_ids], None)


class RootChildrenAccuracy(Accuracy):

    def update_metric(self, y_pred, y_true, in_data):
        root_ids = [i for i in range(in_data.number_of_nodes()) if in_data.out_degree(i) == 0]
        root_ch_id = [i for i in range(in_data.number_of_nodes()) if i not in root_ids and in_data.successors(i).item() in root_ids]
        super(RootChildrenAccuracy, self).update_metric(y_pred[root_ch_id], y_true[root_ch_id], None)


class LeavesAccuracy(Accuracy):

    def update_metric(self, y_pred, y_true, in_data):
        leaves_ids = [i for i in range(in_data.number_of_nodes()) if in_data.in_degrees(i) == 0]
        super(LeavesAccuracy, self).update_metric(y_pred[leaves_ids], y_true[leaves_ids], None)


class MSE(BaseMetric):

    HIGHER_BETTER = False

    def __init__(self):
        super(MSE, self).__init__()
        self.val = 0
        self.n_val = 0

    def update_metric(self, y_pred, y_true, in_data):
        self.val += th.sum((y_pred - y_true).pow(2)).item()
        self.n_val += y_true.size(0)

    def finalise_metric(self):
        self.final_value = self.val / self.n_val


class MAE(BaseMetric):

    HIGHER_BETTER = False

    def __init__(self):
        super(MAE, self).__init__()
        self.val = 0
        self.n_val = 0

    def update_metric(self, y_pred, y_true, in_data):
        self.val += th.abs(th.sum((y_pred - y_true))).item()
        self.n_val += y_true.size(0)

    def finalise_metric(self):
        self.final_value = self.val / self.n_val


class StoreAllMetric(BaseMetric, ABC):

    def __init__(self):
        super(StoreAllMetric, self).__init__()
        self.all_y_pred = None
        self.all_y_true = None

    def update_metric(self, y_pred, y_true, in_data):
        y_pred = copy.deepcopy(y_pred)
        y_true = copy.deepcopy(y_true)

        if self.all_y_pred is None:
            self.all_y_pred = y_pred
        else:
            self.all_y_pred = th.cat((self.all_y_pred, y_pred), dim=0)

        if self.all_y_true is None:
            self.all_y_true = y_true
        else:
            self.all_y_true = th.cat((self.all_y_true, y_true), dim=0)


class Pearson(StoreAllMetric):

    HIGHER_BETTER = True

    def finalise_metric(self):

        vx = self.all_y_pred - th.mean(self.all_y_pred)
        vy = self.all_y_true - th.mean(self.all_y_true)

        cost = th.sum(vx * vy) / (th.sqrt(th.sum(vx ** 2)) * th.sqrt(th.sum(vy ** 2)))
        self.final_value = cost.item()

'''
class OGBGMetric(StoreAllMetric):

    def __init__(self, dataset_name):
        super(OGBGMetric, self).__init__()
        from ogb.graphproppred import Evaluator
        self.evaluator = Evaluator(dataset_name)
'''