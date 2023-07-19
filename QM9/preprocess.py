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


def graph2IDsubgraph(data: PygData,
                     k: int,
                     strategy='neighbor',
                     max_depth=1) -> Tuple[Tensor, Tensor]:
    '''
    data.edge_index (2, M)
    data.edge_attr (M, d)
    data.x (N, d)
    return: (ns, k), (ns, k, k, 1)
    '''
    assert strategy in ['neighbor', 'path', 'subgraph']
    N = data.num_nodes
    adj = SparseTensor(row=data.edge_index[0] + 1,
                       col=data.edge_index[1] + 1,
                       value=data.edge_attr,
                       sparse_sizes=(N + 1, N + 1)).coalesce()  # N+1 实现padding
    rowptr, col, _ = adj.csr()
    # print(adj)
    # print(rowptr)
    # print(col)
    subgs = []
    subadj = []
    if strategy == 'neighbor':
        for i in range(1, N + 1):
            neighbor = col[rowptr[i]:rowptr[i + 1]]
            # print(i)
            # print(neighbor)
            if neighbor.shape[0] == 0:
                # print('case1')
                continue
            elif neighbor.shape[0] < k - 1:
                tsubgs = torch.cat((neighbor,
                                    torch.empty(
                                        (k - 1 - neighbor.shape[0]),
                                        dtype=neighbor.dtype,
                                        device=neighbor.device).fill_(0)),
                                   dim=0).unsqueeze(0)
                # print('case2')
                # print(tsubgs)
            else:
                # print('case3')
                tsubgs = extractsubset(neighbor, k - 1)
                # print(tsubgs)
            tsubgs = torch.cat((torch.empty(
                (tsubgs.shape[0], 1), dtype=tsubgs.dtype,
                device=tsubgs.device).fill_(i), tsubgs),
                               dim=-1)
            # print(i, tsubgs)
            subgs.append(tsubgs)

        # print(subgs)
        subgs = torch.cat(subgs)
        # print(subgs)
        adj = adj.to_dense()
        subadj = extractsubg(adj, subgs)  # (ns, k, k)
        subadj = subadj
        return PygData(subgs=subgs - 1,
                       subadj=subadj,
                       x=data.x,
                       y=data.y,
                       num_subg=torch.tensor((subadj.shape[0]),
                                             dtype=torch.long),
                       num_node=torch.tensor((N), dtype=torch.long))

    elif strategy == 'path':
        # for i in range(1, N + 1):
        #     neighbor = col[rowptr[i]:rowptr[i + 1]]
        #     if neighbor.shape[0] == 0:
        #         continue
        pass

    elif strategy == 'subgraph':
        adj_dense = adj.to_dense()
        for i in range(1, N + 1):
            # print(i)
            close_list = []
            open_list = queue.Queue()
            neighbor = col[rowptr[i]:rowptr[i + 1]]
            all_order_neighbors = []
            all_order_neighbors.append(neighbor.reshape(-1, 1))
            close_list.append(i)
            for index in range(neighbor.shape[0]):
                close_list.append(neighbor[index])

            for depth in range(1, max_depth):
                # print('all_order_neighbor', all_order_neighbors)
                depth_order_neighbor = all_order_neighbors[depth - 1]
                # print('depth_order_neighbor', depth_order_neighbor)
                next_depth_neighbor = []
                for index in range(depth_order_neighbor.shape[0]):
                    open_list.put(depth_order_neighbor[index])
                # print('open_list', open_list)
                while not open_list.empty():
                    current_node = open_list.get()
                    # print('current node', current_node)
                    current_node_neighbor = col[
                        rowptr[current_node]:rowptr[current_node + 1]]
                    # print('current_node_neighbor', current_node_neighbor)
                    for neighbor_index in range(
                            current_node_neighbor.shape[0]):
                        if current_node_neighbor[
                                neighbor_index] not in close_list:
                            close_list.append(
                                current_node_neighbor[neighbor_index])
                            next_depth_neighbor.append(
                                current_node_neighbor[neighbor_index])
                all_order_neighbors.append(
                    torch.tensor(next_depth_neighbor).reshape(-1, 1))

            # print(all_order_neighbors)
            tsubgs = torch.cat(all_order_neighbors, dim=0).reshape(1, -1)
            # print(i, tsubgs)

            if tsubgs.shape[1] == 0:
                # print('case1')
                continue
            elif tsubgs.shape[1] < k - 1:
                # print('case2')
                # print(tsubgs)
                tsubgs = torch.cat((torch.empty(
                    (tsubgs.shape[0], 1),
                    dtype=tsubgs.dtype,
                    device=tsubgs.device).fill_(i), tsubgs),
                                   dim=-1).squeeze(1)
                tsubadj = extractsubg(adj_dense, tsubgs)
                # print(tsubadj)  # [0]
                tsubadj_order = tsubadj + torch.eye(
                    tsubgs.shape[1]).unsqueeze(0).repeat(
                        tsubadj.shape[0], 1, 1)
                deg = min(max_depth, k)
                for order in range(2, deg + 1):
                    tsubadj_order += torch.matrix_power(tsubadj, order)
                # print(tsubadj_order)
                if torch.nonzero(
                        tsubadj_order[0][0]).shape[0] == tsubgs.shape[1]:
                    # print('true')
                    tsubgs = torch.cat((tsubgs[0],
                                        torch.empty(
                                            (k - tsubgs[0].shape[0]),
                                            dtype=tsubgs[0].dtype,
                                            device=tsubgs[0].device).fill_(0)),
                                       dim=0).unsqueeze(0)
                    tsubgs = tsubgs.long()
                    subgs.append(tsubgs)
                    tsubadj = extractsubg(adj_dense, tsubgs)
                    subadj.append(tsubadj)
            else:
                # print('case3')
                tsubgs = extractsubset(tsubgs[0], k - 1)
                tsubgs = tsubgs.long()
                # print(tsubgs)
                tsubgs = torch.cat((torch.empty(
                    (tsubgs.shape[0], 1),
                    dtype=tsubgs.dtype,
                    device=tsubgs.device).fill_(i), tsubgs),
                                   dim=-1).squeeze(1)
                # print(tsubgs)
                tsubadj = extractsubg(adj_dense, tsubgs)
                # print(tsubgs)
                # print(tsubadj)
                tsubadj_order = tsubadj + torch.eye(
                    tsubgs.shape[1]).unsqueeze(0).repeat(
                        tsubadj.shape[0], 1, 1)
                deg = min(max_depth, k)
                for order in range(2, deg + 1):
                    tsubadj_order += torch.matrix_power(tsubadj, order)
                # print(tsubadj_order)
                for ii in range(tsubadj_order.shape[0]):
                    if torch.nonzero(
                            tsubadj_order[ii][0]).shape[0] == tsubgs.shape[1]:
                        # print('true')
                        subgs.append(tsubgs[ii].unsqueeze(0))
                        subadj.append(tsubadj[ii].unsqueeze(0))

            # print(i, tsubgs)
            # tsubadj = extractsubg(adj_dense, tsubgs)
            # print(tsubadj)  # [0]
            # tsubadj_order = tsubadj
            # deg = min(max_depth, k)
            # for order in range(2, deg+1):
            #     tsubadj_order += torch.matrix_power(tsubadj, order)
            # print(tsubadj_order)
            # if torch.nonzero(tsubadj_order).shape[0] == k:
            #     print('true')
            #     subgs.append(tsubgs)
            #     subadj.append(tsubadj)

        # print(subgs)
        subgs = torch.cat(subgs)
        subadj = torch.cat(subadj)
        # print(subadj.shape)
        # adj = adj.to_dense()
        # subadj = extractsubg(adj, subgs)  # (ns, k, k)
        # subadj = subadj
        return PygData(subgs=subgs - 1,
                       subadj=subadj,
                       x=data.x,
                       y=data.y,
                       num_subg=torch.tensor((subadj.shape[0]),
                                             dtype=torch.long),
                       num_node=torch.tensor((N), dtype=torch.long))


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
                            dataset_max_node=29,
                            distance_constrainl=-1,
                            distance_constrainu=-1):
    '''
    data.edge_index (2, M)
    data.edge_attr (M, d)
    data.x (N, d)
    return: adj (1, Nm, Nm), x (1, Nm, d), subgs (ns, k)
    '''
    # print(data.edge_attr)
    N, Nm, d = data.x.shape[0], dataset_max_node, data.x.shape[1]
    # print(data)
    # print(data.x)
    adj = SparseTensor(row=data.edge_index[0],
                       col=data.edge_index[1],
                       value=torch.nonzero(data.edge_attr)[:, 1] + 1,
                       sparse_sizes=(Nm, Nm)).coalesce()
    adj = adj.to_dense().unsqueeze_(0).to(torch.long)
    # print(adj)
    x = torch.cat((data.x, torch.zeros((Nm - N, d),
                                       dtype=data.x.dtype))).unsqueeze_(0)

    # G = nx.Graph()
    # G.clear()
    # for j in range(N):
    #     G.add_node(j, atom=data.x[j])
    # # G.add_nodes_from(np.arange(0, data.num_nodes))
    # # edge_list = []
    # for j in range(data.edge_index.shape[1]):
    #     G.add_edge(data.edge_index[0][j].item(), data.edge_index[1][j].item(), bond=data.edge_attr[j])


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
    if subgs.shape[0] == 0:
        subgs = extractsubset(torch.arange(N), k)
        print(subgs)
    return PygData(adj=adj,
                   x=x,
                   y=data.y,
                   subgs=subgs,
                   pos=data.pos,
                   num_subg=torch.tensor((subgs.shape[0]), dtype=torch.long),
                   num_node=torch.tensor((N), dtype=torch.long))


def graph2IDsubgraph_cluster(data: PygData,
                            k: int,
                            dataset_max_node=29,
                            distance_constrainl=-1,
                            distance_constrainu=-1,
                            resolution=0.5):
    '''
    data.edge_index (2, M)
    data.edge_attr (M, d)
    data.x (N, d)
    return: adj (1, Nm, Nm), x (1, Nm, d), subgs (ns, k)
    '''
    # print(data)
    N, Nm, d = data.x.shape[0], dataset_max_node, data.x.shape[1]
    # print(data)
    # print(data.x)
    # print(data.pos)
    adj = SparseTensor(row=data.edge_index[0],
                       col=data.edge_index[1],
                       value=torch.nonzero(data.edge_attr)[:, 1] + 1,
                       sparse_sizes=(Nm, Nm)).coalesce()
    adj = adj.to_dense().unsqueeze_(0).to(torch.long)
    x = torch.cat((data.x, torch.zeros((Nm - N, d),
                                       dtype=data.x.dtype))).unsqueeze_(0)
    pos = torch.cat((data.pos, torch.zeros((Nm - N, 3),
                                       dtype=data.pos.dtype))).unsqueeze_(0)

    # convert o networkx, current version without node and edge features
    # plt.figure(index, figsize=(20, 20))
    G = nx.Graph()
    G.clear()
    for j in range(N):
        G.add_node(j, atom=data.x[j])
        G.add_node(j, atom=data.x[j])
    # G.add_nodes_from(np.arange(0, data.num_nodes))
    # edge_list = []
    for j in range(data.edge_index.shape[1]):
        G.add_edge(data.edge_index[0][j].item(), data.edge_index[1][j].item(), bond=torch.nonzero(data.edge_attr)[j, 1] + 1.)
        # edge_list.append((data.edge_index[0][j].item(), data.edge_index[1][j].item(), bond=data.edge_attr[j]))
    # G.add_edges_from(edge_list)

    iG = ig.Graph()
    iG = iG.from_networkx(G)
    # print(iG)
    num_clusters = 3 if N < 20 else 4
    # vertex_cluster = iG.community_leading_eigenvector(clusters=num_clusters, weights=data.edge_attr)
    vertex_cluster = iG.community_multilevel(weights='bond', resolution=resolution)
    # print(vertex_cluster)
    # clusters = np.zeros([data.num_nodes])
    # for i in range(len(vertex_cluster)):
    #     # print(vertex_cluster[i])
    #     for each in vertex_cluster[i]:
    #         clusters[each] = i
    #         G.nodes[each]['cluster'] = i

    # clusters = list(clusters)
    # print(clusters)

    # print(vertex_cluster.graph)
    # print(vertex_cluster.modularity)
    # subgraph = vertex_cluster.subgraphs()
    # for i in range(len(subgraph)):
    #     print(subgraph[i])
    # print(vertex_cluster.cluster_graph())

    # nx.draw_networkx(G, node_size=300, node_color=clusters)
    # plt.savefig('graph_figure/' + str(index) + 'test.png')

    subgs = []
    if max([len(vertex_cluster[i]) for i in range(len(vertex_cluster))]) < k:
        subgs = extractsubset(torch.arange(N), k)  # (ns, k)
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
    else:
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
    # print(subgs.shape)
    return PygData(adj=adj,
                   x=x,
                   y=data.y,
                   z=data.z,
                   pos=pos,
                   subgs=subgs,
                   name=data.name,
                   num_subg=torch.tensor((subgs.shape[0]), dtype=torch.long),
                   num_node=torch.tensor((N), dtype=torch.long))