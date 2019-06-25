from .utils import get_new_logger
from collections import namedtuple


# TODO: you MUST to inehirt from torch dataset
class TreeDataset():

    TreeBatch = namedtuple('TreeBatch', ['graph', 'mask', 'x', 'y'])

    def __init__(self, path_dir, file_name_list, name):
        self.data = []
        self.path_dir = path_dir
        self.file_name_list = file_name_list
        self.name = name

        self.logger = get_new_logger('loading.{}'.format(self.name))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_loader(self, batch_size, device, shuffle=False):
        raise NotImplementedError('users must define __load__ to use this base class')



