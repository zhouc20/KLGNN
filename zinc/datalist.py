import imp
from torch_geometric.data import InMemoryDataset

class DataListSet(InMemoryDataset):
    def __init__(self, datalist):
        super().__init__()
        self.data, self.slices = self.collate(datalist)