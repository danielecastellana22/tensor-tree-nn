import os
from abc import abstractmethod, ABC
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from exputils.serialisation import from_pkl_file


class ConstValues:
    UNK = 0
    NO_ELEMENT = -1


class ListDataset(Dataset):

    def __init__(self, list):
        self.data = list

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class BaseDatasetLoader:

    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def get_training_data_loader(self):
        raise NotImplementedError('Must be implemented in subclasses')

    @abstractmethod
    def get_validation_data_loader(self):
        raise NotImplementedError('Must be implemented in subclasses')

    @abstractmethod
    def get_test_data_loader_dict(self):
        raise NotImplementedError('Must be implemented in subclasses')

    @abstractmethod
    def __collate_fun__(self, data):
        raise NotImplementedError('Must be implemented in subclasses')

    def __get_data_loader__(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.__collate_fun__, shuffle=shuffle, num_workers=0)


class FileDatasetLoader(BaseDatasetLoader, ABC):

    def __init__(self, data_dir, max_tr_elements=None, **kwargs):
        super(FileDatasetLoader, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.max_tr_elements = max_tr_elements

    def get_training_data_loader(self):

        train_set = ConcatDataset(self.__load_list_of_datasets__('train').values())

        if self.max_tr_elements:
            train_set = train_set[:self.max_tr_elements]

        return self.__get_data_loader__(train_set, shuffle=True)

    def get_validation_data_loader(self):
        val_set = ConcatDataset(self.__load_list_of_datasets__('validation').values())
        return self.__get_data_loader__(val_set, shuffle=False)

    def get_test_data_loader_dict(self):
        return {k: self.__get_data_loader__(v, shuffle=False) for k, v in self.__load_list_of_datasets__('test').items()}

    def __load_list_of_datasets__(self, tag):
        out_dict = {}
        for f in os.listdir(self.data_dir):
            if tag in f:
                out_dict[f] = ListDataset(from_pkl_file(os.path.join(self.data_dir, f)))
        return out_dict


class OGBGDatasetLoader(BaseDatasetLoader):

    def __init__(self, data_dir, dataset_name, sample_perc=1, **kwargs):
        import torch as th
        from ogb.graphproppred import DglGraphPropPredDataset

        super(OGBGDatasetLoader, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        if sample_perc >= 1 or sample_perc <= 0:
            raise ValueError('Sample percentage must be between 0 and 1')

        if sample_perc <= 1:
            # check if small exists
            small_dataset_path = os.path.join(os.path.join(data_dir, '_'.join(dataset_name.split('-'))),
                                              'small_{:.2f}'.format(sample_perc))
            if os.path.exists(small_dataset_path):
                # just load small dataset
                self.idx_split = th.load(os.path.join(small_dataset_path, 'idx_split.th'))
                self.dataset = th.load(os.path.join(small_dataset_path, 'data.th'))
            else:
                # create the dir
                os.makedirs(small_dataset_path)

                # load everything
                all_dataset = DglGraphPropPredDataset(name=dataset_name, root=data_dir)
                all_idx_split = all_dataset.get_idx_split()

                # create small dataset
                aux_idx_split = {k: v[:int(sample_perc*len(v))] for k,v in all_idx_split.items()}
                idx_to_retain = th.cat(list(aux_idx_split.values())).int()
                all_dataset.graphs = [all_dataset.graphs[i] for i in idx_to_retain]
                all_dataset.labels = [all_dataset.labels[i] for i in idx_to_retain]
                # small_dataset = all_dataset[th.cat(list(aux_idx_split.values()))]
                small_dataset = all_dataset

                prev=0
                small_idx_split = {}
                for k, v in aux_idx_split.items():
                    small_idx_split[k] = th.arange(prev, prev+len(v))
                    prev += len(v)

                # save idx split
                th.save(small_idx_split, os.path.join(small_dataset_path, 'idx_split.th'))
                # save data
                th.save(small_dataset, os.path.join(small_dataset_path, 'data.th'))

                self.idx_split = small_idx_split
                self.dataset = small_dataset

        else:
            # load everything
            self.dataset = DglGraphPropPredDataset(name=dataset_name, root=data_dir)
            self.idx_split = self.dataset.get_idx_split()

    def get_training_data_loader(self):
        idx_train = self.idx_split['train']
        return self.__get_data_loader__(self.dataset[idx_train], shuffle=True)

    def get_validation_data_loader(self):
        idx_valid = self.idx_split['valid']
        return self.__get_data_loader__(self.dataset[idx_valid], shuffle=True)

    def get_test_data_loader_dict(self):
        idx_test = self.idx_split['test']
        return {'test': self.__get_data_loader__(self.dataset[idx_test], shuffle=False)}

    def __collate_fun__(self, data):
        from ogb.graphproppred import collate_dgl
        #label2id = {}
        #mapped_labels = []
        #for label_list in self.dataset.labels:
        #    mapped_labels.append([label2id.setdefault(k, len(label2id)) for k in label_list])
        #self.dataset.labels = mapped_labels
        # TODO: convert feature name and compute label dict
        return collate_dgl(data)


