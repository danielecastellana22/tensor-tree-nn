from torch.utils.data import Dataset


class ListDataset(Dataset):

    def __init__(self, list):
        self.data = list

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
