from dgl.data.utils import load_graphs
from torch_geometric.datasets import ZINC
from torch_geometric.data import Data as PygData
import torch


def load_dataset(dataset: str, task: str = None):
    if dataset in ["dataset1", "dataset2"]:
        assert task is not None, "Please specify a task"
        glist, y = load_graphs(f"data/{dataset}.bin")
        y = y[task]
        with open(f"data/{dataset}_train.txt") as f:
            trn_idx = torch.tensor([int(line) for line in f.readlines()], dtype=torch.long)
        with open(f"data/{dataset}_val.txt") as f:
            val_idx = torch.tensor([int(line) for line in f.readlines()], dtype=torch.long)
        with open(f"data/{dataset}_test.txt") as f:
            tst_idx = torch.tensor([int(line) for line in f.readlines()], dtype=torch.long)
        glist = [
            PygData(edge_index=torch.stack(glist[i].edges(), dim=0),
                    num_nodes=glist[i].number_of_nodes(), y=y[i])
            for i in range(len(glist))
        ]
    elif dataset in ["zinc12", "zinc250", "zinc"]:
        if dataset == 'zinc250':
            ds1 = ZINC("./data", subset=False, split="train")
            ds2 = ZINC("./data", subset=False, split="val")
            ds3 = ZINC("./data", subset=False, split="test")
        else:
            ds1 = ZINC("./data", subset=True, split="train")
            ds2 = ZINC("./data", subset=True, split="val")
            ds3 = ZINC("./data", subset=True, split="test")
        glist = [dat for dat in ds1] + [dat
                                        for dat in ds2] + [dat for dat in ds3]
        idx1, idx2, idx3 = len(ds1), len(ds1) + len(ds2), len(glist)
        trn_idx = torch.arange(idx1)
        val_idx = torch.arange(idx1, idx2)
        tst_idx = torch.arange(idx2, idx3)
    else:
        raise NotImplementedError
    return glist, trn_idx, val_idx, tst_idx