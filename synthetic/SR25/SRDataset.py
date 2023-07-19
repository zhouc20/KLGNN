from torch_geometric.data import InMemoryDataset
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.data.data import Data
import networkx as nx
import numpy as np
from preprocess import graph2IDsubgraph_global_new, graph2IDsubgraph
class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, k=4, distlimitl=-1, distlimitu=3):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        # self.processed_paths = '/data/sr25/processed'
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.k = k
        self.distlimitl = distlimitl
        self.distlimitu = distlimitu

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  # sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(), 1, dtype=torch.float)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1, 0))
            data_list.append(Data(edge_index=edge_index, x=x, y=0, number_of_nodes=datum.number_of_nodes()))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_list = [
            graph2IDsubgraph(dat, 3, 'neighbor', 1)
            # graph2IDsubgraph_global_new(dat, 5, max([g.x.shape[0] for g in data_list]),
            #                             -1, 4)
            for dat in data_list
        ]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def calculate_stats(dataset):
    num_graphs = len(dataset)
    ave_num_nodes = np.array([g.num_nodes for g in dataset]).mean()
    ave_num_edges = np.array([g.num_edges for g in dataset]).mean()
    print(
        f'# Graphs: {num_graphs}, average # nodes per graph: {ave_num_nodes}, average # edges per graph: {ave_num_edges}.')