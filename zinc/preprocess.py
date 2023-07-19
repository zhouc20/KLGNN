import queue
from turtle import distance

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data as PygData
import torch_geometric as pyg
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch import Tensor
from util import extractsubset, extracttuple
import networkx as nx

from scipy.sparse.csgraph import floyd_warshall
import igraph as ig

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


# constrained sampling
def graph2IDsubgraph_global(data: PygData,
                            k: int,
                            dataset_max_node=37,
                            distance_constrainl=-1,
                            distance_constrainu=-1,
                            k_step_rw=20):
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
                       sparse_sizes=(Nm, Nm)).coalesce()
    adj = adj.to_dense().unsqueeze_(0).to(data.edge_attr.dtype)
    x = torch.cat((data.x, torch.zeros((Nm - N, d),
                                       dtype=data.x.dtype))).unsqueeze_(0)

    G = nx.Graph()
    G.clear()
    for j in range(N):
        G.add_node(j, atom=data.x[j])

    for j in range(data.edge_index.shape[1]):
        G.add_edge(data.edge_index[0][j].item(), data.edge_index[1][j].item(), bond=data.edge_attr[j])
    tadj = SparseTensor(row=data.edge_index[0],
                        col=data.edge_index[1],
                        sparse_sizes=(Nm, Nm)).to_dense().numpy()
    distance = floyd_warshall(tadj, directed=False, unweighted=True)
    degree = np.sum(tadj, axis=0).reshape(-1)
    ksteps = np.arange(k_step_rw).tolist()
    rw_landing = get_rw_landing_probs(ksteps, data.edge_index, edge_weight=None,
                                      num_nodes=dataset_max_node, space_dim=0)


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
    return PygData(adj=adj,
                   x=x,
                   y=data.y,
                   subgs=subgs,
                   degree=torch.tensor(degree, dtype=torch.long).unsqueeze(0),
                   distance=torch.tensor(distance, dtype=torch.long).unsqueeze(0),
                   num_subg=torch.tensor((subgs.shape[0]), dtype=torch.long),
                   num_node=torch.tensor((N), dtype=torch.long),
                   rwse=torch.tensor(rw_landing, dtype=torch.float).unsqueeze(0))


# constrained + hierarchical sampling
def graph2IDsubgraph_cluster(data: PygData,
                            k: int,
                            dataset_max_node=37,
                            distance_constrainl=-1,
                            distance_constrainu=-1,
                            resolution=0.5,
                            k_step_rw=20):
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
                       sparse_sizes=(Nm, Nm)).coalesce()
    adj = adj.to_dense().unsqueeze_(0).to(data.edge_attr.dtype)
    x = torch.cat((data.x, torch.zeros((Nm - N, d),
                                       dtype=data.x.dtype))).unsqueeze_(0)

    # convert o networkx, current version without node and edge features

    G = nx.Graph()
    G.clear()
    for j in range(N):
        G.add_node(j, atom=data.x[j])
    for j in range(data.edge_index.shape[1]):
        G.add_edge(data.edge_index[0][j].item(), data.edge_index[1][j].item(), bond=data.edge_attr[j])

    iG = ig.Graph()
    iG = iG.from_networkx(G)
    num_clusters = 3 if N < 20 else 4
    # vertex_cluster = iG.community_leading_eigenvector(clusters=num_clusters, weights=data.edge_attr)
    vertex_cluster = iG.community_multilevel(weights='bond', resolution=resolution)
    tadj = SparseTensor(row=data.edge_index[0],
                        col=data.edge_index[1],
                        sparse_sizes=(Nm, Nm)).to_dense().numpy()
    distance = floyd_warshall(tadj, directed=False, unweighted=True)
    degree = np.sum(tadj, axis=0).reshape(-1)
    ksteps = np.arange(k_step_rw).tolist()
    rw_landing = get_rw_landing_probs(ksteps, data.edge_index, edge_weight=None,
                                      num_nodes=dataset_max_node, space_dim=0)

    subgs = []
    for i in range(len(vertex_cluster)):
        subgs_cluster = extractsubset(torch.tensor(vertex_cluster[i]), k)  #(ns, k)
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
            subgsmask = (subgs_cluster.unsqueeze(-1) * N + subgs_cluster.unsqueeze(-2)).flatten(
                -2, -1)
            subgsmask = tadj[subgsmask].all(dim=-1)
            subgs_cluster = subgs_cluster[subgsmask]
        subgs.append(subgs_cluster)
    subgs = torch.cat(subgs)

    return PygData(adj=adj,
                   x=x,
                   y=data.y,
                   subgs=subgs,
                   degree=torch.tensor(degree, dtype=torch.long).unsqueeze(0),
                   distance=torch.tensor(distance, dtype=torch.long).unsqueeze(0),
                   num_subg=torch.tensor((subgs.shape[0]), dtype=torch.long),
                   num_node=torch.tensor((N), dtype=torch.long),
                   rwse=torch.tensor(rw_landing, dtype=torch.float).unsqueeze(0))


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing