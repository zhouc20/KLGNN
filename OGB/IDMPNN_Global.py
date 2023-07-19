import random
from codecs import ascii_encode
from typing import Final, List, Tuple
from util import cumsum_pad0, deg2rowptr, extracttuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool
from subgraphpooling import AttenPool
from torch_sparse import SparseTensor
from tkinter import _flatten
from torch_geometric.nn import GINEConv


@torch.jit.script
def indexbmm(batch: Tensor, A: Tensor, x: Tensor):
    '''
    A (B, Nm, Nm, d)
    x (ns1+ns2+...+nsB, nm, #perm, d)
    '''
    return torch.einsum("bijd,bjpd->bipd", A[batch], x)


@torch.jit.script
def noindexbmm(A: Tensor, x: Tensor):
    '''
    A (B, Nm, Nm, d)
    x (ns1+ns2+...+nsB, nm, #perm, d)
    '''
    return torch.einsum("bijd,bjpd->bipd", A, x)


def subgs2sparse(subgs: Tensor) -> SparseTensor:
    mask = (subgs >= 0)
    deg = torch.sum(mask, dim=1)
    rowptr = deg2rowptr(deg)
    col = subgs.flatten()[mask.flatten()]
    return SparseTensor(rowptr=rowptr, col=col).device_as(mask)


from math import factorial


class IDMPNN_Global_new(nn.Module):
    num_layer: Final[int]

    def __init__(self,
                 k: int,
                 in_dim: int,
                 hid_dim: int,
                 out_dim,
                 num_layer: int,
                 num_layer_global: int = 0,
                 max_nodez: int = None,
                 max_edgez: int = None,
                 global_pool: str = 'max'):
        super().__init__()

        self.k = k
        self.hid_dim = hid_dim
        self.idemb = nn.Embedding(k + 2, hid_dim)
        self.permdim = factorial(k)
        allperm = extracttuple(torch.arange(k), k)
        self.register_buffer("allperm", allperm.t())  #(k, k!)
        self.num_layer = num_layer
        self.num_layer_global = num_layer_global
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer)
        ])
        self.global_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_global)
        ])
        self.setmlp1 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp2 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.outmlp = nn.Linear(hid_dim, out_dim)
        assert global_pool in ['max', 'add', 'mean', 'attention']
        self.global_pool = eval("global_" + global_pool + "_pool") if global_pool != 'attention' else AttenPool(hid_dim, hid_dim)
        if max_nodez is not None:
            self.inmlp_mod = nn.Embedding(max_nodez + 1, hid_dim)
            self.inmlp = lambda x: self.inmlp_mod(x).squeeze(-2)
        if max_edgez is not None:
            self.edge_emb = nn.Embedding(max_edgez + 1, hid_dim, padding_idx=0)

    def num2batch(self, num_subg: Tensor):
        offset = cumsum_pad0(num_subg)
        # print(offset.shape, num_subg.shape, offset[-1] + num_subg[-1])
        batch = torch.zeros((offset[-1] + num_subg[-1]),
                            device=offset.device,
                            dtype=offset.dtype)
        batch[offset] = 1
        batch[0] = 0
        batch = batch.cumsum_(dim=0)
        return batch

    def forward(self, x: Tensor, subadj: SparseTensor, subgs: Tensor,
                num_subg: Tensor, num_node: Tensor) -> Tensor:
        '''
        Nm = max(Ni)
        x: (B, Nm, d)
        subadj: (B, Nm, Nm)
        subgs: sparse(ns1+ns2+...+nsB, k)
        num_subg: (B) vector of ns
        num_node: (B) vector of N
        '''
        '''
        x (ns1+ns2+...+nsB, N_m, (k-1)!, d)
        '''
        subgs = subgs2sparse(subgs)
        B, Nm = x.shape[0], x.shape[1]
        Bidx = torch.arange(B, device=x.device)
        null_node_mask = torch.zeros((B, Nm+1), device=x.device)
        null_node_mask[Bidx, num_node] = 1
        null_node_mask = null_node_mask.cumsum_(dim=1) > 0
        null_node_mask = null_node_mask[:, :-1]

        subgbatch = self.num2batch(num_subg)  # 依次 ns 个 0, 1, ...

        x = self.inmlp(x)  # (B, Nm, hid_dim)
        x = x[subgbatch].unsqueeze(-2).repeat(
            1, 1, self.permdim, 1)  # (ns1+ns2+...+nsB, Nm, permdim, hid_dim)

        labelsubgptr, labelnodes, _ = subgs.csr()

        subgidx = torch.arange(labelnodes.shape[0], device=labelsubgptr.device)
        subgsize = torch.diff(labelsubgptr)
        cumdeg = torch.zeros_like(subgidx)
        cumdeg[labelsubgptr[1:-1]] = subgsize[:-1]

        cumdeg = cumdeg.cumsum_(dim=0)
        nodeidx = subgidx - cumdeg

        labelbatch = self.num2batch(subgsize)

        x[labelbatch, labelnodes] = x[labelbatch, labelnodes] * self.idemb(
            self.allperm[nodeidx])  # (selectiondim, permdim, hid_dim)
        if subadj.dtype == torch.long:
            subadj = self.edge_emb(subadj)
        else:
            subadj = subadj.unsqueeze_(-1)
        # subadj (B, Nm, Nm, d/1)
        #print(subadj.shape, x.shape)
        subadj = subadj[subgbatch]
        for _ in range(self.num_layer):
            x = x + self.mlps[_](noindexbmm(subadj, x)) #indexbmm(subgbatch, subadj, x))
        # (ns, Nm, (k-1)!, d)
        x[null_node_mask[subgbatch]] = 0  # should not be here?
        # x = self.setmlp1(x.sum(dim=1))  # (ns, (k-1)!, d)
        # x = self.setmlp2(x.mean(dim=1))  # (ns, d)
        x = self.setmlp1(x.mean(dim=2))  # (ns, Nm, d)
        for _ in range(self.num_layer_global):
            x = x + self.global_mlps[_](torch.einsum("bijd,bjd->bjd", subadj, x))  # (ns, Nm, d)
        x = self.setmlp2(x.sum(dim=1))  # (ns, d)
        x = self.global_pool(x, subgbatch)
        return self.outmlp(x)


class IDMPNN_Global_parallel(nn.Module):
    num_layer: Final[int]

    def __init__(self,
                 k: int,
                 in_dim: int,
                 hid_dim: int,
                 out_dim,
                 num_layer: int,
                 num_layer_global: int = 0,
                 num_layer_id: int = 1,
                 num_layer_regression: int = 1,
                 max_nodez: int = None,
                 max_edgez1: int = None,
                 max_edgez2: int = None,
                 max_edgez3: int = None,
                 node_pool: str = 'mean',
                 subgraph_pool: str = 'add',
                 global_pool: str = 'max',
                 rate: float = 1.0,
                 cat: str = 'add',
                 drop_ratio: float = 0.0,
                 drop_perm: float = 1.0,
                 norm_type: str = 'layer',
                 ensemble_test: bool = False):
        super().__init__()

        self.k = k
        self.hid_dim = hid_dim
        self.rate = rate
        self.ensemble_test = ensemble_test
        self.drop_ratio = drop_ratio
        self.idemb = nn.Embedding(k + 2, hid_dim)
        self.permdim = factorial(k)
        self.drop_permdim = int(self.permdim * drop_perm)
        allperm = extracttuple(torch.arange(k), k)
        self.register_buffer("allperm", allperm.t())  # (k, k!)
        self.num_layer = num_layer
        self.num_layer_global = num_layer_global
        self.num_layer_id = num_layer_id
        self.num_layer_regression = num_layer_regression
        assert cat in ['add', 'hadamard_product', 'cat', 'none']
        self.cat = cat
        hid_dim_global = 2 * hid_dim if cat == 'cat' else hid_dim
        assert norm_type in ['layer', 'batch']
        self.norm_type = norm_type
        self.graph_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer)
        ])
        self.graph_mlps_aggregate = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer)
        ])
        self.id_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_id)
        ])
        self.id_mlps_aggregate = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_id)
        ])
        self.global_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_global)
        ])
        self.setmlp1 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp2 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp3 = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_regression)
        ])
        self.setmlp4 = nn.Sequential(nn.Linear(hid_dim_global, hid_dim_global), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim_global, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True))
        self.outmlp = nn.Sequential(nn.Linear(hid_dim, out_dim), nn.Sigmoid())
        assert node_pool in ['max', 'add', 'mean']
        assert subgraph_pool in ['max', 'add', 'mean']
        assert global_pool in ['max', 'add', 'mean', 'attention', 'mix']
        self.node_pool_type = node_pool
        self.node_pool = eval("global_" + node_pool + "_pool")
        self.subgraph_pool_type = subgraph_pool
        self.subgraph_pool = eval("global_" + subgraph_pool + "_pool")
        self.global_pool_type = global_pool
        self.global_pool = eval("global_" + global_pool + "_pool") if global_pool in ['max', 'add', 'mean'] else AttenPool(hid_dim, hid_dim)
        # self.inmlp_mod = nn.Embedding(max_nodez + 1, hid_dim)
        # self.inmlp1 = lambda x: self.inmlp_mod(x).squeeze(-2)
        # self.inmlp2 = nn.Sequential(nn.Linear(in_dim-1, hid_dim), nn.ReLU(inplace=True),
        #                   nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True))
        self.inmlp = DiscreteAtomEncoder(self.hid_dim, 9, max_nodez + 1)
        self.edge_emb1 = nn.Embedding(max_edgez1 + 1, hid_dim, padding_idx=0)
        self.edge_emb2 = nn.Embedding(max_edgez2 + 1, hid_dim, padding_idx=0)
        self.edge_emb3 = nn.Embedding(max_edgez3 + 1, hid_dim, padding_idx=0)

    def num2batch(self, num_subg: Tensor):
        offset = cumsum_pad0(num_subg)
        # print(offset.shape, num_subg.shape, offset[-1] + num_subg[-1])
        batch = torch.zeros((offset[-1] + num_subg[-1]),
                            device=offset.device,
                            dtype=offset.dtype)
        batch[offset] = 1
        batch[0] = 0
        batch = batch.cumsum_(dim=0)
        return batch

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, subgs: Tensor,
                num_subg: Tensor, num_node: Tensor, num_edge: Tensor) -> Tensor:
        '''
        Nm = max(Ni)
        x: (B, Nm, d)
        adj: (B, Nm, Nm)
        subgs: sparse(ns1+ns2+...+nsB, k)
        num_subg: (B) vector of ns
        num_node: (B) vector of N
        '''
        '''
        x (ns1+ns2+...+nsB, N_m, (k-1)!, d)
        '''
        subgs = subgs2sparse(subgs)
        B = num_node.shape[0]
        Nm = torch.max(num_node)
        padded_x = []
        padded_adj = []
        for b in range(B):
            padded_x.append(torch.cat((x[torch.sum(num_node[:b]): torch.sum(num_node[:(b+1)])], torch.zeros((Nm - num_node[b], x.shape[1]),
                                       dtype=x.dtype, device=x.device))).unsqueeze(0))
            adj = SparseTensor(
                row=edge_index[0, torch.sum(num_edge[:b]): torch.sum(num_edge[:b + 1])] - torch.sum(num_node[:b]),
                col=edge_index[1, torch.sum(num_edge[:b]): torch.sum(num_edge[:b + 1])] - torch.sum(num_node[:b]),
                value=edge_attr[torch.sum(num_edge[:b]): torch.sum(num_edge[:b + 1]), :],
                sparse_sizes=(Nm, Nm)).coalesce()
            padded_adj.append(adj.to_dense().unsqueeze(0).to(torch.long))
        x = torch.cat(padded_x)
        adj = torch.cat(padded_adj).to(x.device)
        Bidx = torch.arange(B, device=x.device)
        null_node_mask = torch.zeros((B, Nm+1), device=x.device)
        null_node_mask[Bidx, num_node] = 1
        null_node_mask = null_node_mask.cumsum_(dim=1) > 0
        null_node_mask = null_node_mask[:, :-1]

        subgbatch = self.num2batch(num_subg)  # 依次 ns 个 0, 1, ...

        # x1 = self.inmlp1(x[:, :, 0].to(torch.long))  # (B, Nm, hid_dim)
        # x2 = self.inmlp2(x[:, :, 1:].to(torch.float))
        # x = x1 + x2
        x = self.inmlp(x)
        z = torch.ones(x.shape[0:2], device=x.device, dtype=torch.long) * (self.k + 1)
        z = self.idemb(z)  # (B, Nm, hid_dim)
        if self.training:
            z = z[subgbatch].unsqueeze(-2).repeat(
                1, 1, self.drop_permdim, 1)  # (ns1+ns2+...+nsB, Nm, permdim, hid_dim)
        else:
            z = z[subgbatch].unsqueeze(-2).repeat(
                1, 1, self.permdim, 1)

        labelsubgptr, labelnodes, _ = subgs.csr()

        subgidx = torch.arange(labelnodes.shape[0], device=labelsubgptr.device)
        subgsize = torch.diff(labelsubgptr)
        cumdeg = torch.zeros_like(subgidx)
        cumdeg[labelsubgptr[1:-1]] = subgsize[:-1]

        cumdeg = cumdeg.cumsum_(dim=0)
        nodeidx = subgidx - cumdeg

        labelbatch = self.num2batch(subgsize)

        select_perm = random.sample(range(self.permdim), self.drop_permdim) if self.training else range(self.permdim)
        perm = self.allperm[:, select_perm]
        z[labelbatch, labelnodes] = self.idemb(perm[nodeidx])  # (selectiondim, permdim, hid_dim)

        adj_1 = self.edge_emb1(adj[:, :, :, 0])  # subadj (B, Nm, Nm, d/1)
        adj_2 = self.edge_emb2(adj[:, :, :, 1])
        adj_3 = self.edge_emb3(adj[:, :, :, 2])
        adj = adj_1 + adj_2 + adj_3
        for _ in range(self.num_layer):
            x = x + self.graph_mlps[_](torch.einsum("bijd,bjd->bid", adj, x))  # (B, Nm, d)
            x = F.dropout(x, self.drop_ratio, self.training)
        x = x[subgbatch]  # (ns, Nm, d)
        adj_ = adj[subgbatch]  # subadj (ns, Nm, Nm, d/1)

        # if self.training:
        ns = 0
        trn_idx = []
        for b in range(B):
            if num_subg[b] == 0:
                continue
            sample_rate = self.rate if (Nm <= 100 or not self.training) else self.rate / 5
            n = int(sample_rate * num_subg[b]) + 1 if (self.training or self.ensemble_test or Nm > 100) else num_subg[b]
            trn_idx.append(random.sample(range(ns, ns + num_subg[b]), n))
            ns += num_subg[b]
        # trn_idx = random.sample(range(ns), int(self.rate * ns))
        trn_idx = list(_flatten(trn_idx))

        # if not self.training:
        #     trn_idx = list(range(len(subgbatch)))
        z = z[trn_idx]
        adj_ = adj_[trn_idx]
        x = x[trn_idx]
        for _ in range(self.num_layer_id):
            z = z + self.id_mlps[_](noindexbmm(adj_, z))  # (ns, Nm, k!, d)
            z = F.dropout(z, self.drop_ratio, self.training)
        z[null_node_mask[subgbatch][trn_idx]] = 0  # set virtual nodes to zero
        z = self.setmlp1(z.mean(dim=2))  # (ns, Nm, d)
        if self.cat == 'add':
            x = x + z
        elif self.cat == 'hadamard_product':
            x = x * z
        elif self.cat == 'cat':
            x = torch.cat([x, z], dim=-1)
        else:
            x = x
        x = self.setmlp4(x)
        for _ in range(self.num_layer_global):
            x = x + self.global_mlps[_](torch.einsum("bijd,bjd->bid", adj_, x))  # (ns, Nm, d)
            x = F.dropout(x, self.drop_ratio, self.training)
        x[null_node_mask[subgbatch][trn_idx]] = 0
        subgidx = torch.arange(subgbatch[trn_idx].size(0), device=x.device)
        subgnode_batch = subgidx.reshape(-1, 1).repeat(1, Nm).reshape(-1)
        x = self.setmlp2(self.subgraph_pool(x.reshape(-1, self.hid_dim), subgnode_batch))  # (ns, d)
        if self.global_pool_type == 'mix':
            x = self.global_pool(x, subgbatch[trn_idx]) + global_max_pool(x, subgbatch[trn_idx])
        else:
            x = self.global_pool(x, subgbatch[trn_idx])
        for _ in range(self.num_layer_regression):
            x = x + self.setmlp3[_](x)  # (B, d)
        return self.outmlp(x)
        # else:
        #     for _ in range(self.num_layer_id):
        #         z = z + self.id_mlps[_](noindexbmm(subadj_, z))  # (ns, Nm, (k-1)!, d)
        #     z[null_node_mask[subgbatch]] = 0  # should not be here?
        #     z = self.setmlp1(z.mean(dim=2))  # (ns, Nm, d)
        #     x = x + z
        #     for _ in range(self.num_layer_global):
        #         x = x + self.global_mlps[_](torch.einsum("bijd,bjd->bjd", subadj_, x))  # (ns, Nm, d)
        #     x = self.setmlp2(x.sum(dim=1))  # (ns, d)
        #     if self.global_pool_type == 'mix':
        #         x = self.global_pool(x, subgbatch) + global_max_pool(x, subgbatch)
        #     else:
        #         x = self.global_pool(x, subgbatch)
        #     return self.outmlp(x)


class IDMPNN_Transformer(nn.Module):
    num_layer: Final[int]

    def __init__(self,
                 k: int,
                 in_dim: int,
                 hid_dim: int,
                 out_dim,
                 num_layer: int,
                 num_layer_global: int = 0,
                 num_layer_id: int = 1,
                 num_layer_regression: int = 1,
                 max_nodez: int = None,
                 max_edgez1: int = None,
                 max_edgez2: int = None,
                 max_edgez3: int = None,
                 node_pool: str = 'mean',
                 subgraph_pool: str = 'add',
                 global_pool: str = 'max',
                 rate: float = 1.0,
                 cat: str = 'add',
                 drop_ratio: float = 0.0,
                 drop_perm: float = 1.0,
                 norm_type: str = 'layer',
                 ensemble_test: bool = False,
                 final_concat: str = 'none',
                 num_head: int = 8):
        super().__init__()

        self.k = k
        self.hid_dim = hid_dim
        self.rate = rate
        self.ensemble_test = ensemble_test
        self.drop_ratio = drop_ratio
        self.idemb = nn.Embedding(k + 2, hid_dim)
        self.permdim = factorial(k)
        self.drop_permdim = int(self.permdim * drop_perm)
        allperm = extracttuple(torch.arange(k), k)
        self.register_buffer("allperm", allperm.t())  # (k, k!)
        self.num_layer = num_layer
        self.num_layer_global = num_layer_global
        self.num_layer_id = num_layer_id
        self.num_layer_regression = num_layer_regression
        assert cat in ['add', 'hadamard_product', 'cat', 'none']
        self.cat = cat
        assert final_concat in ['add', 'hadamard_product', 'cat', 'none']
        self.final_concat = final_concat
        hid_dim_global = 2 * hid_dim if cat == 'cat' else hid_dim
        hid_dim_final = 2 * hid_dim if final_concat == 'cat' else hid_dim
        assert norm_type in ['layer', 'batch']
        self.norm_type = norm_type
        self.graph_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer)
        ])
        self.graph_ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, 2*hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(2*hid_dim, hid_dim), nn.LayerNorm(hid_dim)) for _ in range(num_layer)
        ])
        self.id_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_id)
        ])
        self.global_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_global)
        ])
        self.global_ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, 2 * hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(2 * hid_dim, hid_dim), nn.LayerNorm(hid_dim)) for _ in range(num_layer_global)
        ])
        self.graph_self_attn = nn.ModuleList([torch.nn.MultiheadAttention(
            self.hid_dim, num_head, dropout=drop_ratio, batch_first=True) for _ in range(num_layer)])
        self.global_self_attn = nn.ModuleList([torch.nn.MultiheadAttention(
            self.hid_dim, num_head, dropout=drop_ratio, batch_first=True) for _ in range(num_layer_global)])
        self.graph_attn_norm = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(num_layer)])
        self.global_attn_norm = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(num_layer_global)])
        self.setmlp1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp3 = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_regression)
        ])
        self.setmlp4 = nn.Sequential(nn.Linear(hid_dim_global, hid_dim_global), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim_global, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim_global),
                          nn.ReLU(inplace=True))
        self.outmlp = nn.Sequential(nn.Linear(hid_dim_final, out_dim))
        assert node_pool in ['max', 'add', 'mean']
        assert subgraph_pool in ['max', 'add', 'mean']
        assert global_pool in ['max', 'add', 'mean', 'attention', 'mix']
        self.node_pool_type = node_pool
        self.node_pool = eval("global_" + node_pool + "_pool")
        self.subgraph_pool_type = subgraph_pool
        self.subgraph_pool = eval("global_" + subgraph_pool + "_pool")
        self.global_pool_type = global_pool
        self.global_pool = eval("global_" + global_pool + "_pool") if global_pool in ['max', 'add', 'mean'] else AttenPool(hid_dim, hid_dim)
        self.inmlp_mod = DiscreteAtomEncoder(self.hid_dim, 9, max_nodez + 1)
        # self.inmlp1 = lambda x: self.inmlp_mod(x).squeeze(-2)
        # self.inmlp2 = nn.Sequential(nn.Linear(in_dim - 1, hid_dim), nn.ReLU(inplace=True),
        #                             nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True))
        self.edge_emb1 = nn.Embedding(max_edgez1 + 1, hid_dim, padding_idx=0)
        self.edge_emb2 = nn.Embedding(max_edgez2 + 1, hid_dim, padding_idx=0)
        self.edge_emb3 = nn.Embedding(max_edgez3 + 1, hid_dim, padding_idx=0)

    def num2batch(self, num_subg: Tensor):
        offset = cumsum_pad0(num_subg)
        # print(offset.shape, num_subg.shape, offset[-1] + num_subg[-1])
        batch = torch.zeros((offset[-1] + num_subg[-1]),
                            device=offset.device,
                            dtype=offset.dtype)
        batch[offset] = 1
        batch[0] = 0
        batch = batch.cumsum_(dim=0)
        return batch

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, subgs: Tensor,
                num_subg: Tensor, num_node: Tensor, num_edge: Tensor) -> Tensor:
        '''
        Nm = max(Ni)
        x: (B, Nm, d)
        adj: (B, Nm, Nm)
        subgs: sparse(ns1+ns2+...+nsB, k)
        num_subg: (B) vector of ns
        num_node: (B) vector of N
        '''
        '''
        x (ns1+ns2+...+nsB, N_m, (k-1)!, d)
        '''
        subgs = subgs2sparse(subgs)
        B = num_node.shape[0]
        Nm = torch.max(num_node)
        padded_x = []
        padded_adj = []
        for b in range(B):
            padded_x.append(torch.cat(
                (x[torch.sum(num_node[:b]): torch.sum(num_node[:(b + 1)])], torch.zeros((Nm - num_node[b], x.shape[1]),
                                                                                        dtype=x.dtype,
                                                                                        device=x.device))).unsqueeze(0))
            adj = SparseTensor(
                row=edge_index[0, torch.sum(num_edge[:b]): torch.sum(num_edge[:b + 1])] - torch.sum(num_node[:b]),
                col=edge_index[1, torch.sum(num_edge[:b]): torch.sum(num_edge[:b + 1])] - torch.sum(num_node[:b]),
                value=edge_attr[torch.sum(num_edge[:b]): torch.sum(num_edge[:b + 1]), :],
                sparse_sizes=(Nm, Nm)).coalesce()
            padded_adj.append(adj.to_dense().unsqueeze(0).to(torch.long))
        x = torch.cat(padded_x)
        adj = torch.cat(padded_adj).to(x.device)
        Bidx = torch.arange(B, device=x.device)
        null_node_mask = torch.zeros((B, Nm + 1), device=x.device)
        null_node_mask[Bidx, num_node] = 1
        null_node_mask = null_node_mask.cumsum_(dim=1) > 0
        null_node_mask = null_node_mask[:, :-1]

        subgbatch = self.num2batch(num_subg)  # 依次 ns 个 0, 1, ...

        # x1 = self.inmlp1(x[:, :, 0].to(torch.long))  # (B, Nm, hid_dim)
        # x2 = self.inmlp2(x[:, :, 1:].to(torch.float))
        # x = x1 + x2
        x = self.inmlp_mod(x)

        z = torch.ones(x.shape[0:2], device=x.device, dtype=torch.long) * (self.k + 1)
        z = self.idemb(z)  # (B, Nm, hid_dim)
        if self.training:
            z = z[subgbatch].unsqueeze(-2).repeat(
                1, 1, self.drop_permdim, 1)  # (ns1+ns2+...+nsB, Nm, permdim, hid_dim)
        else:
            z = z[subgbatch].unsqueeze(-2).repeat(
                1, 1, self.permdim, 1)

        labelsubgptr, labelnodes, _ = subgs.csr()

        subgidx = torch.arange(labelnodes.shape[0], device=labelsubgptr.device)
        subgsize = torch.diff(labelsubgptr)
        cumdeg = torch.zeros_like(subgidx)
        cumdeg[labelsubgptr[1:-1]] = subgsize[:-1]

        cumdeg = cumdeg.cumsum_(dim=0)
        nodeidx = subgidx - cumdeg

        labelbatch = self.num2batch(subgsize)

        select_perm = random.sample(range(self.permdim), self.drop_permdim) if self.training else range(self.permdim)
        perm = self.allperm[:, select_perm]
        z[labelbatch, labelnodes] = self.idemb(perm[nodeidx])  # (selectiondim, permdim, hid_dim)

        adj_1 = self.edge_emb1(adj[:, :, :, 0])  # subadj (B, Nm, Nm, d/1)
        adj_2 = self.edge_emb2(adj[:, :, :, 1])
        adj_3 = self.edge_emb3(adj[:, :, :, 2])
        adj = adj_1 + adj_2 + adj_3
        for _ in range(self.num_layer):
            x1 = self.graph_mlps[_](torch.einsum("bijd,bjd->bid", adj, x))  # (B, Nm, d)
            x1 = F.dropout(x1, self.drop_ratio, self.training)
            # if not self.training:
            #     print(x.shape, null_node_mask.shape)
            x2 = self.graph_self_attn[_](x, x, x,
                               attn_mask=None,
                               key_padding_mask=null_node_mask,
                               need_weights=False)[0]
            x2 = self.graph_attn_norm[_](x2)
            # self.attn_weights = A.detach().cpu()
            # x2 = F.dropout(x2, self.drop_ratio, self.training)
            x = x + x1 + x2
            x = x + self.graph_ffn[_](x)
        pre_x = x.sum(dim=1)
        #x = x[subgbatch]  # (ns, Nm, d)
        adj_ = adj[subgbatch]  # subadj (ns, Nm, Nm, d/1)

        # if self.training:
        ns = 0
        trn_idx = []
        for b in range(B):
            if num_subg[b] == 0:
                continue
            n = int(self.rate * num_subg[b]) + 1 if (self.training or self.ensemble_test) else num_subg[b]
            trn_idx.append(random.sample(range(ns, ns + num_subg[b]), n))
            ns += num_subg[b]
        # trn_idx = random.sample(range(ns), int(self.rate * ns))
        trn_idx = list(_flatten(trn_idx))

        # if not self.training:
        #     trn_idx = list(range(len(subgbatch)))
        z = z[trn_idx]
        adj_ = adj_[trn_idx]
        #x = x[trn_idx]
        for _ in range(self.num_layer_id):
            z = z + self.id_mlps[_](noindexbmm(adj_, z))  # (ns, Nm, k!, d)
            z = F.dropout(z, self.drop_ratio, self.training)
        z[null_node_mask[subgbatch][trn_idx]] = 0  # set virtual nodes to zero
        z = self.setmlp1(z.mean(dim=2))  # (ns, Nm, d)
        z = self.setmlp2(self.subgraph_pool(z.reshape(-1, Nm * self.hid_dim), subgbatch[trn_idx]).reshape(B, Nm, self.hid_dim))  # (B, Nm, d)
        if self.cat == 'add':
            x = x + z
        elif self.cat == 'hadamard_product':
            x = x * z
        elif self.cat == 'cat':
            x = torch.cat([x, z], dim=-1)
        else:
            x = x
        x = self.setmlp4(x)
        for _ in range(self.num_layer_global):
            x1 = self.global_mlps[_](torch.einsum("bijd,bjd->bid", adj, x))  # (B, Nm, d)
            x1 = F.dropout(x1, self.drop_ratio, self.training)
            x2 = self.global_self_attn[_](x, x, x,
                                         attn_mask=None,
                                         key_padding_mask=null_node_mask,
                                         need_weights=False)[0]
            # self.attn_weights = A.detach().cpu()
            x2 = self.global_attn_norm[_](x2)
            # x2 = F.dropout(x2, self.drop_ratio, self.training)
            x = x + x1 + x2
            x = x + self.global_ffn[_](x)
        x[null_node_mask] = 0
        graphidx = torch.arange(B, device=x.device)
        subgnode_batch = graphidx.reshape(-1, 1).repeat(1, Nm).reshape(-1)
        x = self.setmlp2(self.global_pool(x.reshape(-1, self.hid_dim), subgnode_batch))  # (B, d)
        for _ in range(self.num_layer_regression):
            x = x + self.setmlp3[_](x)  # (B, d)
        if self.final_concat == 'cat':
            x = torch.cat([x, pre_x], dim=-1)
        elif self.final_concat == 'add':
            x = x + pre_x
        elif self.final_concat == 'hadamard_product':
            x = x * pre_x
        return self.outmlp(x)



class DiscreteAtomEncoder(nn.Module):
    def __init__(self, hidden_channels, max_num_features=10,
                 max_num_values=500, padding=False):  # 10, change it for correctly counting number of parameters
        super().__init__()
        if not padding:
            self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels)
                                         for _ in range(max_num_features)])
        else:
            self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels, padding_idx=0)
                                             for _ in range(max_num_features)])

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(2)
        out = 0
        for i in range(x.size(2)):
            out = out + self.embeddings[i](x[:, :, i])
        return out


class IDMPNN_Discrete(nn.Module):
    num_layer: Final[int]

    def __init__(self,
                 k: int,
                 in_dim: int,
                 hid_dim: int,
                 out_dim,
                 num_layer: int,
                 num_layer_global: int = 0,
                 num_layer_id: int = 1,
                 num_layer_regression: int = 1,
                 max_nodez: int = None,
                 max_edgez: int = None,
                 node_pool: str = 'mean',
                 subgraph_pool: str = 'add',
                 global_pool: str = 'max',
                 rate: float = 1.0,
                 cat: str = 'add',
                 drop_ratio: float = 0.0,
                 drop_perm: float = 1.0,
                 norm_type: str = 'layer',
                 ensemble_test: bool = False):
        super().__init__()

        self.k = k
        self.hid_dim = hid_dim
        self.rate = rate
        self.ensemble_test = ensemble_test
        self.drop_ratio = drop_ratio
        self.idemb = nn.Embedding(k + 2, hid_dim)
        self.permdim = factorial(k)
        self.drop_permdim = int(self.permdim * drop_perm)
        allperm = extracttuple(torch.arange(k), k)
        self.register_buffer("allperm", allperm.t())  # (k, k!)
        self.num_layer = num_layer
        self.num_layer_global = num_layer_global
        self.num_layer_id = num_layer_id
        self.num_layer_regression = num_layer_regression
        assert cat in ['add', 'hadamard_product', 'cat', 'none']
        self.cat = cat
        hid_dim_global = 2 * hid_dim if cat == 'cat' else hid_dim
        assert norm_type in ['layer', 'batch']
        self.norm_type = norm_type
        self.graph_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer)
        ])
        self.graph_edge_emb = nn.ModuleList([
            DiscreteBondEncoder(self.hid_dim, 3, max_edgez + 1) for _ in range(num_layer)
        ])
        self.id_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_id)
        ])
        self.id_edge_emb = nn.ModuleList([
            DiscreteBondEncoder(self.hid_dim, 3, max_edgez + 1) for _ in range(num_layer_id)
        ])
        self.global_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_global)
        ])
        self.global_edge_emb = nn.ModuleList([
            DiscreteBondEncoder(self.hid_dim, 3, max_edgez + 1) for _ in range(num_layer_global)
        ])
        self.setmlp1 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp2 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp3 = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_regression)
        ])
        self.setmlp4 = nn.Sequential(nn.Linear(hid_dim_global, hid_dim_global), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim_global, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True))
        self.outmlp = nn.Sequential(nn.Linear(hid_dim, out_dim), nn.Sigmoid())
        assert node_pool in ['max', 'add', 'mean']
        assert subgraph_pool in ['max', 'add', 'mean']
        assert global_pool in ['max', 'add', 'mean', 'attention', 'mix']
        self.node_pool_type = node_pool
        self.node_pool = eval("global_" + node_pool + "_pool")
        self.subgraph_pool_type = subgraph_pool
        self.subgraph_pool = eval("global_" + subgraph_pool + "_pool")
        self.global_pool_type = global_pool
        self.global_pool = eval("global_" + global_pool + "_pool") if global_pool in ['max', 'add', 'mean'] else AttenPool(hid_dim, hid_dim)
        self.inmlp = DiscreteAtomEncoder(self.hid_dim, 9, max_nodez + 1)

    def num2batch(self, num_subg: Tensor):
        offset = cumsum_pad0(num_subg)
        # print(offset.shape, num_subg.shape, offset[-1] + num_subg[-1])
        batch = torch.zeros((offset[-1] + num_subg[-1]),
                            device=offset.device,
                            dtype=offset.dtype)
        batch[offset] = 1
        batch[0] = 0
        batch = batch.cumsum_(dim=0)
        return batch

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, subgs: Tensor,
                num_subg: Tensor, num_node: Tensor, num_edge: Tensor) -> Tensor:
        '''
        Nm = max(Ni)
        x: (B, Nm, d)
        adj: (B, Nm, Nm)
        subgs: sparse(ns1+ns2+...+nsB, k)
        num_subg: (B) vector of ns
        num_node: (B) vector of N
        '''
        '''
        x (ns1+ns2+...+nsB, N_m, (k-1)!, d)
        '''
        subgs = subgs2sparse(subgs)
        B = num_node.shape[0]
        Nm = torch.max(num_node)
        padded_x = []
        padded_adj = []
        for b in range(B):
            padded_x.append(torch.cat((x[torch.sum(num_node[:b]): torch.sum(num_node[:(b+1)])], torch.zeros((Nm - num_node[b], x.shape[1]),
                                       dtype=x.dtype, device=x.device))).unsqueeze(0))
            # print(edge_index[0, torch.sum(num_edge[:b]): torch.sum(num_edge[:b+1])] - torch.sum(num_node[:b]))
            # print(Nm)
            # print(edge_attr[torch.sum(num_edge[:b]): torch.sum(num_edge[:b+1]), :])
            adj = SparseTensor(row=edge_index[0, torch.sum(num_edge[:b]): torch.sum(num_edge[:b+1])] - torch.sum(num_node[:b]),
                               col=edge_index[1, torch.sum(num_edge[:b]): torch.sum(num_edge[:b+1])] - torch.sum(num_node[:b]),
                               value=edge_attr[torch.sum(num_edge[:b]): torch.sum(num_edge[:b+1]), :],
                               sparse_sizes=(Nm, Nm)).coalesce()
            padded_adj.append(adj.to_dense().unsqueeze(0).to(torch.long))
        x = torch.cat(padded_x)
        adj = torch.cat(padded_adj).to(x.device)
        # print(adj.shape)
        # print(adj[1])
        Bidx = torch.arange(B, device=x.device)
        null_node_mask = torch.zeros((B, Nm+1), device=x.device)
        null_node_mask[Bidx, num_node] = 1
        null_node_mask = null_node_mask.cumsum_(dim=1) > 0
        null_node_mask = null_node_mask[:, :-1]

        subgbatch = self.num2batch(num_subg)  # 依次 ns 个 0, 1, ...

        x = self.inmlp(x)
        z = torch.ones(x.shape[0:2], device=x.device, dtype=torch.long) * (self.k + 1)
        z = self.idemb(z)  # (B, Nm, hid_dim)
        if self.training:
            z = z[subgbatch].unsqueeze(-2).repeat(
                1, 1, self.drop_permdim, 1)  # (ns1+ns2+...+nsB, Nm, permdim, hid_dim)
        else:
            z = z[subgbatch].unsqueeze(-2).repeat(
                1, 1, self.permdim, 1)

        labelsubgptr, labelnodes, _ = subgs.csr()

        subgidx = torch.arange(labelnodes.shape[0], device=labelsubgptr.device)
        subgsize = torch.diff(labelsubgptr)
        cumdeg = torch.zeros_like(subgidx)
        cumdeg[labelsubgptr[1:-1]] = subgsize[:-1]

        cumdeg = cumdeg.cumsum_(dim=0)
        nodeidx = subgidx - cumdeg

        labelbatch = self.num2batch(subgsize)

        select_perm = random.sample(range(self.permdim), self.drop_permdim) if self.training else range(self.permdim)
        perm = self.allperm[:, select_perm]
        z[labelbatch, labelnodes] = self.idemb(perm[nodeidx])  # (selectiondim, permdim, hid_dim)

        for _ in range(self.num_layer):
            x = x + self.graph_mlps[_](torch.einsum("bijd,bjd->bid", self.graph_edge_emb[_](adj), x))  # (B, Nm, d)
            x = F.dropout(x, self.drop_ratio, self.training)
        x = x[subgbatch]  # (ns, Nm, d)
        adj_ = adj[subgbatch]  # subadj (ns, Nm, Nm, d/1)

        # if self.training:
        ns = 0
        trn_idx = []
        for b in range(B):
            if num_subg[b] == 0:
                continue
            sample_rate = self.rate if Nm < 100 else self.rate / 5
            n = int(sample_rate * num_subg[b]) + 1 if (self.training or self.ensemble_test) else num_subg[b]
            trn_idx.append(random.sample(range(ns, ns + num_subg[b]), n))
            ns += num_subg[b]
        # trn_idx = random.sample(range(ns), int(self.rate * ns))
        trn_idx = list(_flatten(trn_idx))

        # if not self.training:
        #     trn_idx = list(range(len(subgbatch)))
        z = z[trn_idx]
        adj_ = adj_[trn_idx]
        x = x[trn_idx]
        for _ in range(self.num_layer_id):
            z = z + self.id_mlps[_](noindexbmm(self.id_edge_emb[_](adj_), z))  # (ns, Nm, k!, d)
            z = F.dropout(z, self.drop_ratio, self.training)
        z[null_node_mask[subgbatch][trn_idx]] = 0  # set virtual nodes to zero
        z = self.setmlp1(z.mean(dim=2))  # (ns, Nm, d)
        if self.cat == 'add':
            x = x + z
        elif self.cat == 'hadamard_product':
            x = x * z
        elif self.cat == 'cat':
            x = torch.cat([x, z], dim=-1)
        else:
            x = x
        x = self.setmlp4(x)
        for _ in range(self.num_layer_global):
            x = x + self.global_mlps[_](torch.einsum("bijd,bjd->bid", self.global_edge_emb[_](adj_), x))  # (ns, Nm, d)
            x = F.dropout(x, self.drop_ratio, self.training)
        x[null_node_mask[subgbatch][trn_idx]] = 0
        subgidx = torch.arange(subgbatch[trn_idx].size(0), device=x.device)
        subgnode_batch = subgidx.reshape(-1, 1).repeat(1, Nm).reshape(-1)
        x = self.setmlp2(self.subgraph_pool(x.reshape(-1, self.hid_dim), subgnode_batch))  # (ns, d)
        if self.global_pool_type == 'mix':
            x = self.global_pool(x, subgbatch[trn_idx]) + global_max_pool(x, subgbatch[trn_idx])
        else:
            x = self.global_pool(x, subgbatch[trn_idx])
        for _ in range(self.num_layer_regression):
            x = x + self.setmlp3[_](x)  # (B, d)
        return self.outmlp(x)


class DiscreteBondEncoder(nn.Module):
    def __init__(self, hidden_channels, max_num_features=10,
                 max_num_values=500, padding=True):  # 10, change it for correctly counting number of parameters
        super().__init__()
        if not padding:
            self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels)
                                         for _ in range(max_num_features)])
        else:
            self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels, padding_idx=0)
                                             for _ in range(max_num_features)])

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(3)
        out = 0
        for i in range(x.size(3)):
            out = out + self.embeddings[i](x[:, :, :, i])
        return out



class ID_GINE(nn.Module):
    num_layer: Final[int]

    def __init__(self,
                 k: int,
                 in_dim: int,
                 hid_dim: int,
                 out_dim,
                 num_layer: int,
                 num_layer_global: int = 0,
                 num_layer_id: int = 1,
                 num_layer_regression: int = 1,
                 max_nodez: int = None,
                 max_edgez1: int = None,
                 max_edgez2: int = None,
                 max_edgez3: int = None,
                 node_pool: str = 'mean',
                 subgraph_pool: str = 'add',
                 global_pool: str = 'max',
                 rate: float = 1.0,
                 cat: str = 'add',
                 drop_ratio: float = 0.0,
                 drop_perm: float = 1.0,
                 norm_type: str = 'layer',
                 ensemble_test: bool = False):
        super().__init__()

        self.k = k
        self.hid_dim = hid_dim
        self.rate = rate
        self.ensemble_test = ensemble_test
        self.drop_ratio = drop_ratio
        self.idemb = nn.Embedding(k + 2, hid_dim)
        self.permdim = factorial(k)
        self.drop_permdim = int(self.permdim * drop_perm)
        allperm = extracttuple(torch.arange(k), k)
        self.register_buffer("allperm", allperm.t())  # (k, k!)
        self.num_layer = num_layer
        self.num_layer_global = num_layer_global
        self.num_layer_id = num_layer_id
        self.num_layer_regression = num_layer_regression
        assert cat in ['add', 'hadamard_product', 'cat', 'none']
        self.cat = cat
        hid_dim_global = 2 * hid_dim if cat == 'cat' else hid_dim
        assert norm_type in ['layer', 'batch']
        self.norm_type = norm_type
        self.graph_mlps = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True))) for _ in range(num_layer)
        ])
        self.graph_mlps_aggregate = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True))) for _ in range(num_layer)
        ])
        self.id_mlps = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True))) for _ in range(num_layer_id)
        ])
        self.id_mlps_aggregate = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True))) for _ in range(num_layer_id)
        ])
        self.global_mlps = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True))) for _ in range(num_layer_global)
        ])
        self.setmlp1 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp2 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp3 = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_regression)
        ])
        self.setmlp4 = nn.Sequential(nn.Linear(hid_dim_global, hid_dim_global), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim_global, hid_dim), nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim),
                          nn.ReLU(inplace=True))
        self.outmlp = nn.Sequential(nn.Linear(hid_dim, out_dim), nn.Sigmoid())
        assert node_pool in ['max', 'add', 'mean']
        assert subgraph_pool in ['max', 'add', 'mean']
        assert global_pool in ['max', 'add', 'mean', 'attention', 'mix']
        self.node_pool_type = node_pool
        self.node_pool = eval("global_" + node_pool + "_pool")
        self.subgraph_pool_type = subgraph_pool
        self.subgraph_pool = eval("global_" + subgraph_pool + "_pool")
        self.global_pool_type = global_pool
        self.global_pool = eval("global_" + global_pool + "_pool") if global_pool in ['max', 'add', 'mean'] else AttenPool(hid_dim, hid_dim)
        if max_nodez is not None:
            self.inmlp_mod = nn.Embedding(max_nodez + 1, hid_dim)
            self.inmlp = lambda x: self.inmlp_mod(x).squeeze(-2)
        if max_edgez1 is not None:
            self.edge_emb1 = nn.Embedding(max_edgez1 + 1, hid_dim, padding_idx=0)
            self.edge_emb2 = nn.Embedding(max_edgez2 + 1, hid_dim, padding_idx=0)
            self.edge_emb3 = nn.Embedding(max_edgez3 + 1, hid_dim, padding_idx=0)

    def num2batch(self, num_subg: Tensor):
        offset = cumsum_pad0(num_subg)
        # print(offset.shape, num_subg.shape, offset[-1] + num_subg[-1])
        batch = torch.zeros((offset[-1] + num_subg[-1]),
                            device=offset.device,
                            dtype=offset.dtype)
        batch[offset] = 1
        batch[0] = 0
        batch = batch.cumsum_(dim=0)
        return batch

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, subgs: Tensor,
                num_subg: Tensor, num_node: Tensor, num_edge: Tensor) -> Tensor:
        '''
        Nm = max(Ni)
        x: (\sum Ni, d)
        adj: (B, Nm, Nm)
        subgs: sparse(ns1+ns2+...+nsB, k)
        num_subg: (B) vector of ns
        num_node: (B) vector of N
        '''
        '''
        x (ns1+ns2+...+nsB, N_m, (k-1)!, d)
        '''
        B = num_subg.shape[0]
        x = self.inmlp(x)  # (B*Nm, hid_dim)
        x_pre = x.reshape(B, -1, self.hid_dim)  # (B, Nm, hid_dim)
        Nm = x_pre.shape[1]
        subgs = subgs2sparse(subgs)
        Bidx = torch.arange(B, device=x.device)
        null_node_mask = torch.zeros((B, Nm+1), device=x.device)
        null_node_mask[Bidx, num_node] = 1
        null_node_mask = null_node_mask.cumsum_(dim=1) > 0
        null_node_mask = null_node_mask[:, :-1]

        subgbatch = self.num2batch(num_subg)  # 依次 ns 个 0, 1, ...
        # subg_perm_batch = self.num2batch(num_subg * self.permdim)  # ns*permdim 个 0,1...
        # z = x_pre[subgbatch].unsqueeze(-2).repeat(1, 1, self.permdim, 1)  # (ns, Nm, permdim, d) whether overlap with x

        z = torch.ones(x_pre.shape[0:2], device=x.device, dtype=torch.long) * (self.k + 1)
        z = self.idemb(z)  # (B, Nm, hid_dim)
        if self.training:
            z = z[subgbatch].unsqueeze(-2).repeat(
                1, 1, self.drop_permdim, 1)  # (ns1+ns2+...+nsB, Nm, permdim, hid_dim)
        else:
            z = z[subgbatch].unsqueeze(-2).repeat(
                1, 1, self.permdim, 1)

        labelsubgptr, labelnodes, _ = subgs.csr()

        subgidx = torch.arange(labelnodes.shape[0], device=labelsubgptr.device)
        subgsize = torch.diff(labelsubgptr)
        cumdeg = torch.zeros_like(subgidx)
        cumdeg[labelsubgptr[1:-1]] = subgsize[:-1]

        cumdeg = cumdeg.cumsum_(dim=0)
        nodeidx = subgidx - cumdeg

        labelbatch = self.num2batch(subgsize)

        select_perm = random.sample(range(self.permdim), self.drop_permdim) if self.training else range(self.permdim)
        perm = self.allperm[:, select_perm]
        z[labelbatch, labelnodes] = self.idemb(perm[nodeidx])  # (selectiondim, permdim, hid_dim)

        ns = 0
        trn_idx = []
        select_ns = []
        for b in range(B):
            if num_subg[b] == 0:
                continue
            n = int(self.rate * num_subg[b]) + 1 if (self.training or self.ensemble_test) else num_subg[b]
            trn_idx.append(random.sample(range(ns, ns + num_subg[b]), n))
            ns += num_subg[b]
            select_ns.append(n)
        select_ns = torch.cat(select_ns)
        trn_idx = list(_flatten(trn_idx))
        z = z[trn_idx]
        z = z.permute(0, 2, 1, 3).reshape(-1, self.hid_dim)   # (\sum ns*perm_dim*Nm, hid_dim)
        bond1 = self.edge_emb1(edge_attr[:, 0]) + self.edge_emb2(edge_attr[:, 1]) + self.edge_emb3(edge_attr[:, 2]) # (\sum Ei, d)

        for _ in range(self.num_layer):
            x = x + self.graph_mlps[_](x, edge_index, bond1)  # (B*Nm, d)
            x = F.dropout(x, self.drop_ratio, self.training)
        x = x.reshape(B, Nm, self.hid_dim)[subgbatch[trn_idx]]  # (ns, Nm, d)
        # bond2 = self.edge_emb2(edge_attr)

        ne = 0
        accumulate_offset = 0
        new_edge_index = []
        new_edge_attr = []
        for b in range(B):
            this_edge = edge_index[:, ne: ne + num_edge[b]]
            ne += num_edge[b]
            this_edge = this_edge.repeat(1, select_ns[b] * self.permdim)
            this_edge += accumulate_offset
            accumulate_offset += (select_ns[b] * self.permdim - 1) * Nm
            this_offset = torch.ones([select_ns[b] * self.permdim], dtype=torch.long, device=x.device) * num_edge[b]
            offset = (self.num2batch(this_offset) * Nm).unsqueeze(0).repeat(2, 1)
            this_edge += offset
            new_edge_index.append(this_edge)
            new_edge_attr.append(bond1[ne, ne + num_edge[b], :].repeat(self.permdim * select_ns[b], 1))
        edge_index_2 = torch.cat(new_edge_index, dim=1)
        bond2 = torch.cat(new_edge_attr)

        for _ in range(self.num_layer_id):
            z = z + self.id_mlps[_](z, edge_index_2, bond2)  # (ns*perm*Nm, d)
            z = F.dropout(z, self.drop_ratio, self.training)
        z = z.reshape(-1, self.permdim, Nm, self.hid_dim).permute(0, 2, 1, 3)  # (ns, Nm, perm, d)
        z[null_node_mask[subgbatch][trn_idx]] = 0  # set virtual nodes to zero
        z = self.setmlp1(z.mean(dim=2))  # (ns, Nm, d)
        if self.cat == 'add':
            x = x + z
        elif self.cat == 'hadamard_product':
            x = x * z
        elif self.cat == 'cat':
            x = torch.cat([x, z], dim=-1)
        else:
            x = x
        x = self.setmlp4(x).reshape(-1, self.hid_dim)  # (ns * Nm, d)
        ne = 0
        accumulate_offset = 0
        new_edge_index = []
        new_edge_attr = []
        for b in range(B):
            this_edge = edge_index[:, ne: ne + num_edge[b]]
            ne += num_edge[b]
            this_edge = this_edge.repeat(1, select_ns[b])
            this_edge += accumulate_offset
            accumulate_offset += (select_ns[b] - 1) * Nm
            this_offset = torch.ones([select_ns[b]], dtype=torch.long, device=x.device) * num_edge[b]
            offset = (self.num2batch(this_offset) * Nm).unsqueeze(0).repeat(2, 1)
            this_edge += offset
            new_edge_index.append(this_edge)
            new_edge_attr.append(bond1[ne, ne + num_edge[b], :].repeat(select_ns[b], 1))
        edge_index_3 = torch.cat(new_edge_index, dim=1)
        bond3 = torch.cat(new_edge_attr)
        for _ in range(self.num_layer_global):
            x = x + self.global_mlps[_](x, edge_index_3, bond3)  # (ns*Nm, d)
            x = F.dropout(x, self.drop_ratio, self.training)
        x = x.reshape(-1, Nm, self.hid_dim)
        x[null_node_mask[subgbatch][trn_idx]] = 0
        subgidx = torch.arange(subgbatch[trn_idx].size(0), device=x.device)
        subgnode_batch = subgidx.reshape(-1, 1).repeat(1, Nm).reshape(-1)
        x = self.setmlp2(self.subgraph_pool(x.reshape(-1, self.hid_dim), subgnode_batch))  # (ns, d)
        if self.global_pool_type == 'mix':
            x = self.global_pool(x, subgbatch[trn_idx]) + global_max_pool(x, subgbatch[trn_idx])
        else:
            x = self.global_pool(x, subgbatch[trn_idx])
        for _ in range(self.num_layer_regression):
            x = x + self.setmlp3[_](x)  # (B, d)
        return self.outmlp(x)
