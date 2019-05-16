from abc import ABC, abstractmethod
from .utils import get_new_logger
import os

class TreeDataset(ABC):

    def __init__(self, path_dir, file_name):
        self.trees = []
        self.path_dir = path_dir
        self.file_name = file_name

        self.logger = get_new_logger('loading.{}'.format(os.path.join(path_dir,file_name)))

    def __getitem__(self, idx):
        return self.trees[idx]

    def __len__(self):
        return len(self.trees)

    @abstractmethod
    def __load_trees__(self):
        raise NotImplementedError('users must define __load__ to use this base class')

    @abstractmethod
    def get_loader(self, batch_size, device, shuffle=False):
        raise NotImplementedError('users must define __load__ to use this base class')

