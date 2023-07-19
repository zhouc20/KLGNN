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


def graph2IDsubgraph_global_new(data: PygData,
                            k: int,
                            dataset_max_node=37,
                            distance_constrainl=-1,
                            distance_constrainu=-1):
    '''
    data.edge_index (2, M)
    data.edge_attr (M, d)
    data.x (N, d)
    return: adj (1, Nm, Nm), x (1, Nm, d), subgs (ns, k)
    '''
    N, Nm, d = data.x.shape[0], dataset_max_node, data.x.shape[1]
    adj = SparseTensor(row=data.edge_index[0],
                       col=data.edge_index[1],
                       # value=data.edge_attr,
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
    subadj = extractsubg(adj, subgs)
    return PygData(subadj=subadj,
                   adj=adj,
                   x=x,
                   y=data.y,
                   subgs=subgs,
                   num_subg=torch.tensor((subgs.shape[0]), dtype=torch.long),
                   num_node=torch.tensor((N), dtype=torch.long))


def graph2IDsubgraph(data: PygData, k: int, strategy='bfs', max_depth=1, dataset_max_node=35):
    '''
    data.edge_index (2, M)
    data.edge_attr (M, d)
    data.x (N, d)
    return: (ns, N), (ns, N, N, 1)
    '''
    assert strategy in ['neighbor', 'path', 'subgraph', 'bfs', 'bfs_new']
    N = data.num_nodes
    adj = SparseTensor(row=data.edge_index[0] + 1,
                       col=data.edge_index[1] + 1,
                       value=torch.ones(data.edge_index.size(1)),
                       sparse_sizes=(N + 1, N + 1)).coalesce()  # N+1 实现padding
    rowptr, col, _ = adj.csr()
    # print(adj)
    # print(rowptr)
    # print(col)
    subgs = []
    subadj = []
    num_permute_nodes = []
    if strategy == 'neighbor':
        for i in range(1, 2):
            neighbor = col[rowptr[i]:rowptr[i + 1]]
            # neighbor = torch.arange(2, 26)
            if neighbor.shape[0] == 0:
                continue
            elif neighbor.shape[0] <= k - 1:
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
        # subgs = torch.cat([torch.zeros([24, 1]), torch.arange(1, 25).unsqueeze(1)], dim=-1).to(torch.long) + 1
        # print(subgs.shape)
        adj = adj.to_dense()
        subadj = extractsubg(adj, subgs)  # (ns, k, k)
        print(subgs)
        return PygData(subgs=subgs - 1,
                       subadj=subadj,
                       adj=adj[1:, 1:],
                       x=data.x.unsqueeze(0),
                       y=data.y,
                       num_subg=torch.tensor((subgs.shape[0]), dtype=torch.long),
                       num_node=torch.tensor((N), dtype=torch.long))

    elif strategy == 'bfs':
        adj_dense = adj.to_dense()
        for i in range(1):
        # for i in range(1, N + 1):
            adapt_hop = max_depth
            # for hop in range(1, N):
            #     subset, edge_index, mapping, edge_mask = pyg.utils.k_hop_subgraph(
            #         i, hop, data.edge_index + 1, relabel_nodes=False)
            #     if subset.shape[0] >= k:
            #         adapt_hop = hop
            #         break
            subset, edge_index, mapping, edge_mask = pyg.utils.k_hop_subgraph(
                i, adapt_hop, data.edge_index + 1, relabel_nodes=False)
            # print(subset)
            tsubgs = subset.tolist()
            for idx in range(subset.shape[0]):
                if tsubgs[idx] == i:
                    del tsubgs[idx]
                    break
            tsubgs = torch.tensor(tsubgs, dtype=torch.long)
            tsubgs = extractsubset(tsubgs, k - 1)
            tsubgs = tsubgs.long()
            # print(tsubgs)
            tsubgs = torch.cat((torch.empty(
                (tsubgs.shape[0], 1), dtype=tsubgs.dtype,
                device=tsubgs.device).fill_(i), tsubgs),
                               dim=-1).squeeze(1)
            # print(tsubgs)
            for j in range(tsubgs.shape[0]):
                perm_graph_edge, _ = pyg.utils.subgraph(tsubgs[j], edge_index)
                # print(perm_graph_edge)
                if not pyg.utils.contains_isolated_nodes(perm_graph_edge, num_nodes=k):
                    # print('true')
                    # tsubgs_list = tsubgs[j].tolist()
                    # for node in range(1, N + 1):
                    #     if node not in tsubgs_list:
                    #         tsubgs_list.append(node)
                    # tsubgs_list = torch.tensor(tsubgs_list, dtype=torch.long, device=tsubgs.device)
                    # tsubgs_list = torch.cat((tsubgs_list,
                    #                 torch.empty((dataset_max_node - tsubgs_list.shape[0]),
                    #                             dtype=tsubgs_list.dtype,
                    #                             device=tsubgs_list.device).fill_(0)),
                    #                 dim=0).unsqueeze(0)
                    subgs.append(tsubgs[j].unsqueeze(0))

        subgs = torch.cat(subgs)  # (ns, Nm)
        # print(subgs)
        subadj = extractsubg(adj_dense, subgs)  # (ns, Nm, Nm)

        return PygData(subgs=subgs - 1,
                       subadj=subadj,
                       adj=adj[1:, 1:],
                       x=torch.tensor(data.x, dtype=torch.float),
                       y=data.y,
                       num_subg=torch.tensor((subgs.shape[0]), dtype=torch.long),
                       num_node=torch.tensor((N), dtype=torch.long))
