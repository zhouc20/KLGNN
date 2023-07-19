import queue
from turtle import distance

import torch
from torch import Tensor
from typing import Tuple
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data as PygData
import torch_geometric as pyg
from torch import Tensor
from util import extractsubset, extracttuple

from scipy.sparse.csgraph import floyd_warshall

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


def graph2IDsubgraph_global(data: PygData,
                            k: int,
                            dataset_max_node=37,
                            distance_constrain=-1):
    '''
    data.edge_index (2, M)
    data.edge_attr (M, d)
    data.x (N, d)
    return: adj (1, Nm, Nm), x (1, Nm, d), subgs (ns, k)
    '''
    N, Nm, d = data.x.shape[0], dataset_max_node, data.x.shape[1]
    adj = SparseTensor(row=data.edge_index[0],
                       col=data.edge_index[1],
                       value=data.edge_attr,
                       sparse_sizes=(Nm, Nm)).coalesce()  # N+1 实现padding
    adj = adj.to_dense().unsqueeze_(0).to(data.edge_attr.dtype)
    x = torch.cat((data.x, torch.zeros((Nm - N, d), dtype=data.x.dtype))).unsqueeze_(0)
    subgs = extractsubset(torch.arange(N), k) #(ns, k)
    if distance_constrain > 0:
        tadj = SparseTensor(row=data.edge_index[0],
                       col=data.edge_index[1],
                       sparse_sizes=(N, N)).to_dense().numpy()
        tadj = floyd_warshall(tadj, directed=False, unweighted=True)
        tadj = torch.from_numpy(tadj) <= distance_constrain
        tadj = tadj.flatten()
        subgsmask = (subgs.unsqueeze(-1)*N + subgs.unsqueeze(-2)).flatten(-2, -1)
        subgsmask = tadj[subgsmask].all(dim=-1)
        subgs = subgs[subgsmask]
    return PygData(subadj=adj,
                   x=x,
                   y=data.y,
                   subgs=subgs,
                   num_subg=torch.tensor((subgs.shape[0]), dtype=torch.long),
                   num_node=torch.tensor((N), dtype=torch.long))


def graph2IDsubgraph_global_new(data: PygData,
                            k: int,
                            dataset_max_node=56,
                            distance_constrainl=-1,
                            distance_constrainu=-1,
                            key_dim=32):
    '''
    data.edge_index (2, M)
    data.edge_attr (M, d)
    data.x (N, d)
    return: adj (1, Nm, Nm), x (1, Nm, d), subgs (ns, k)
    '''
    N, Nm, d = data.x.shape[0], data.x.shape[0], data.x.shape[1]
    print(data)
    adj = SparseTensor(row=data.edge_index[0],
                       col=data.edge_index[1],
                       value=torch.ones(data.edge_index.size(1)),
                       sparse_sizes=(Nm, Nm)).coalesce()
    adj = adj.to_dense().unsqueeze_(0)
    x = torch.cat((data.x, torch.zeros((Nm - N, d),
                                       dtype=data.x.dtype))).unsqueeze_(0)
    subgs = extractsubset(torch.arange(N), k)  #(ns, k)
    if distance_constrainu >= 0 or distance_constrainl >= 0:
        tadj = SparseTensor(row=data.edge_index[0],
                            col=data.edge_index[1],
                            sparse_sizes=(N, N)).to_dense().numpy()
        tadj = floyd_warshall(tadj, directed=False, unweighted=True)
        tadj = torch.from_numpy(tadj)
        mask = torch.ones_like(tadj, dtype=torch.bool)
        if distance_constrainu >= 0:
            mask &= tadj <= distance_constrainu
        if distance_constrainl >= 0:
            mask &= tadj >= distance_constrainl
        mask |= torch.eye(N, dtype=torch.bool)
        tadj = mask.flatten()
        subgsmask = (subgs.unsqueeze(-1) * N + subgs.unsqueeze(-2)).flatten(
            -2, -1)
        subgsmask = tadj[subgsmask].all(dim=-1)
        subgs = subgs[subgsmask]
    pad_adj = SparseTensor(row=data.edge_index[0] + 1,
                       col=data.edge_index[1] + 1,
                       value=torch.ones(data.edge_index.size(1)),
                       sparse_sizes=(Nm+1, Nm+1)).coalesce()
    pad_adj = pad_adj.to_dense().unsqueeze_(0)
    subadj = extractsubg(adj, subgs)
    # subadj = extractsubg(pad_adj, subgs+1)
    # print(data.y)
    # print(data.pos)
    return PygData(adj=adj,
                   subadj=subadj,
                   x=x,
                   y=data.y,
                   pos=data.pos,
                   subgs=subgs,
                   num_subg=torch.tensor((subgs.shape[0]), dtype=torch.long),
                   num_node=torch.tensor((N), dtype=torch.long),
                   key=torch.randn(key_dim))