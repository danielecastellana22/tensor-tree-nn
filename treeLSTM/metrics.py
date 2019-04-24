from abc import ABC, abstractmethod
import torch as th

class BaseMetric(ABC):

    def __init__(self):
        self.final_value = None

    def get_value(self):
        return self.final_value

    @abstractmethod
    def update_metric(self, out, batch):
        raise NotImplementedError('users must define update_metrics to use this base class')

    @abstractmethod
    def finalise_metric(self):
        raise NotImplementedError('users must define finalise_metric to use this base class')


class LabelAccuracy(BaseMetric):

    def __init__(self):
        BaseMetric.__init__(self)
        self.n_nodes = 0
        self.n_correct = 0

    def update_metric(self, out, batch):
        # root_ids = [i for i in range(batch.graph.number_of_nodes()) if batch.graph.out_degree(i) == 0]
        # leaves_ids = [i for i in range(batch.graph.number_of_nodes()) if batch.graph.in_degree(i) == 0]
        pred = th.argmax(out, 1)
        self.n_correct += th.sum(th.eq(batch.label, pred)).item()
        self.n_nodes += len(batch.label)
        # dev_accs.append([acc, len(batch.label)])
        # root_acc = np.sum(batch.label.cpu().data.numpy()[root_ids] == pred.cpu().data.numpy()[root_ids])
        # dev_root_accs.append([root_acc, len(root_ids)])

    def finalise_metric(self):
        self.final_value = self.n_correct / self.n_nodes
