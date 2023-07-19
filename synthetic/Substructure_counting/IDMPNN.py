from typing import Final, List, Tuple
import torch
import torch_geometric
import torch_sparse
import torch.nn as nn
from torch_geometric.data import Data as PygData
from torch import Tensor
from torch_sparse import SparseTensor


def extracttuple(elems: Tensor, k: int) -> Tensor:
    '''
    extract k-tuple from set
    elems : (N)
    '''
    def removedegeneratedtuple(tuples: Tensor):
        '''
        tuples (N, k)
        '''
        sorted_tuple = torch.sort(tuples, dim=-1)[0]
        mask = (torch.diff(sorted_tuple, dim=-1) > 0).all(dim=-1)
        return tuples[mask]

    N = elems.shape[0]
    elemslist: List[Tensor] = []
    for i in range(k):
        shapevec = [1] * (k)
        shapevec[i] = -1
        elemslist.append(elems.reshape(shapevec))
        shapevec = [N] * k
        elemslist[-1] = elemslist[-1].expand(shapevec)
    tuples = torch.stack(elemslist, dim=-1).reshape(-1, k)
    return removedegeneratedtuple(tuples)


def extractsubset(elems: Tensor, k: int) -> Tensor:
    '''
    extract k-tuple from set
    elems : (N)
    '''
    N = elems.shape[0]
    elemslist: List[Tensor] = []
    for i in range(k):
        shapevec = [1] * (k)
        shapevec[i] = -1
        elemslist.append(elems.reshape(shapevec))
        shapevec = [N] * k
        elemslist[-1] = elemslist[-1].expand(shapevec)
    tuples = torch.stack(elemslist, dim=-1).reshape(-1, k)
    mask = (torch.diff(tuples, dim=-1) > 0).all(dim=-1)
    return tuples[mask]


def extractsubg(padded_adj: Tensor, subgs: Tensor) -> Tensor:
    '''
    padded_adj (N+1, N+1)
    subgs (ns, k)
    non-exists edge will be 0
    '''
    N, k = padded_adj.shape[0], subgs.shape[-1]
    subgs = N * subgs.unsqueeze(1) + subgs.unsqueeze(2).expand(-1, k, k)
    subgs = subgs.flatten(-2, -1)
    subadj = padded_adj.flatten()[subgs].reshape(-1, k, k)
    return subadj


def graph2IDsubgraph(data: PygData, k: int) -> Tuple[Tensor, Tensor]:
    '''
    data.edge_index (2, M)
    data.x (N, d)
    return: (ns, k), (ns, k, k)
    '''
    x: Tensor = data.x
    N = data.num_nodes
    adj = SparseTensor(row=data.edge_index[0] + 1,
                       col=data.edge_index[1] + 1,
                       sparse_sizes=(N + 1, N + 1)).coalesce()  # N+1 实现padding
    rowptr, col, _ = adj.csr()
    subgs = []
    for i in range(1, N + 1):
        neighbor = col[rowptr[i]:rowptr[i + 1]]
        if neighbor.shape[0] == 0:
            continue
        elif neighbor.shape[0] < k - 1:
            tsubgs = torch.cat((neighbor,
                                torch.empty((k - 1 - neighbor.shape[0]),
                                            dtype=neighbor.dtype,
                                            device=neighbor.device).fill_(0)),
                               dim=0).unsqueeze(0)
        else:
            tsubgs = extractsubset(neighbor, k - 1)
        tsubgs = torch.cat((torch.empty(
            (tsubgs.shape[0], 1), dtype=tsubgs.dtype,
            device=tsubgs.device).fill_(i), tsubgs),
                           dim=-1)
        subgs.append(tsubgs)
    subgs = torch.cat(subgs)
    adj = adj.to_dense()
    subadj = extractsubg(adj, subgs)  # (ns, k, k)
    return PygData(subgs = subgs - 1, subadj=subadj)


from torch_geometric.nn.glob import global_add_pool


class IDMPNN(nn.Module):
    '''
        This is a local version of IDMPNN, i.e. localized 1,k-WL.
        The message passing range is the labeled subgraph instead of full graph.
        This does not affect substructure counting power, but is more efficient.
    '''
    num_layer: Final[int]

    def __init__(self, k: int, in_dim: int, hid_dim: int, out_dim,
                 num_layer: int):
        super().__init__()

        self.idemb = nn.Embedding(k, hid_dim)
        allperm = extracttuple(torch.arange(k - 1) + 1, k - 1)
        allperm = torch.cat((torch.zeros(
            (allperm.shape[0], 1), dtype=allperm.dtype), allperm),
                            dim=-1) #((k-1)!, k)
        self.register_buffer("allperm", allperm)
        self.num_layer = num_layer
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer)
        ])
        self.setmlp1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True))
        self.setmlp2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True))
        self.outmlp = nn.Linear(hid_dim, out_dim)
        self.inmlp = nn.Linear(in_dim, hid_dim)

    def forward(self,
                x: Tensor,
                subadj: Tensor,
                subgs: Tensor,
                xoffset=None,
                subgbatch=None) -> Tensor:
        '''
        x: (N, d)
        subadj: (ns, k, k)
        subgs: (ns, k)
        '''
        if x is not None:
            x = self.inmlp(x)
            if subgbatch is not None:
                x = x[subgs +
                    xoffset[subgbatch].unsqueeze(-1)]  #(ns, k, d) x不用padd
            else:
                x = x[subgs]
            x = x.unsqueeze(1) * self.idemb(self.allperm).unsqueeze(0) #(ns, (k-1)!, k, d)
        else:
            ns = subgs.shape[0]
            x = self.idemb(self.allperm).unsqueeze(0).expand(ns, -1, -1, -1)
        subadj = subadj.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        for _ in range(self.num_layer):
            x = x + self.mlps[_](subadj @ x)
        x = x.transpose(1, 2) #(ns, k, (k-1)!, d)
        x[subgs<0] = 0
        x = self.setmlp1(x.sum(dim=1)) #(ns, (k-1)!, d)
        x = self.setmlp2(x.mean(dim=1)) #(ns, d)
        if subgbatch is None:
            x = x.sum(dim=0)
        else:
            x = global_add_pool(x, subgbatch)
        return self.outmlp(x)


if __name__ == '__main__':
    import torch
    print(torch.__version__)
