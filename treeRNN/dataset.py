from experiments.execution_utils import get_sub_logger
from collections import namedtuple
from torch.utils.data import Dataset


class TreeDataset(Dataset):

    TreeBatch = namedtuple('TreeBatch', ['graph', 'mask', 'x', 'y'])

    def __init__(self, path_dir, file_name_list, name):
        Dataset.__init__(self)
        self.data = []
        self.path_dir = path_dir
        self.file_name_list = file_name_list
        self.name = name

        self.logger = get_sub_logger(self.name)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    # TODO: maybe this should be an external function in the utils file. Hence, we can use ConcatDataset
    def get_loader(self, batch_size, device, shuffle=False):
        raise NotImplementedError('users must define __load__ to use this base class')
