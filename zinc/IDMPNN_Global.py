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


# Architecture (a) in paper; jointly learn features and ID
class IDMPNN_Global(nn.Module):
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
        subadj = subadj[subgbatch]
        for _ in range(self.num_layer):
            x = x + self.mlps[_](noindexbmm(subadj, x))  # indexbmm(subgbatch, subadj, x))
        # (ns, Nm, (k-1)!, d)
        x[null_node_mask[subgbatch]] = 0
        # x = self.setmlp1(x.sum(dim=1))  # (ns, (k-1)!, d)
        # x = self.setmlp2(x.mean(dim=1))  # (ns, d)
        x = self.setmlp1(x.mean(dim=2))  # (ns, Nm, d)
        for _ in range(self.num_layer_global):
            x = x + self.global_mlps[_](torch.einsum("bijd,bjd->bjd", subadj, x))  # (ns, Nm, d)
        x = self.setmlp2(x.sum(dim=1))  # (ns, d)
        x = self.global_pool(x, subgbatch)
        return self.outmlp(x)


# Architecture (b) in paper; learn feature and IDs parallel
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
                 max_edgez: int = None,
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
                 rw_step: int = 20,
                 se_dim: int = 0,
                 se_type: str = 'linear'):
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
        if max_nodez is not None:
            self.inmlp_mod = nn.Embedding(max_nodez + 1, hid_dim - se_dim)
            self.inmlp = lambda x: self.inmlp_mod(x).squeeze(-2)
        if max_edgez is not None:
            self.edge_emb = nn.Embedding(max_edgez + 1, hid_dim, padding_idx=0)
        assert se_type in ['linear', 'mlp']
        self.se_dim = se_dim
        if se_type == 'linear':
            self.se_emb = nn.Linear(rw_step, se_dim)
        else:
            self.se_emb = nn.Sequential(nn.Linear(rw_step, 2 * se_dim), nn.ReLU(inplace=True),
                                        nn.Linear(2 * se_dim, 2 * se_dim), nn.ReLU(inplace=True),
                                        nn.Linear(2 * se_dim, se_dim))

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

    def forward(self, x: Tensor, adj: SparseTensor, subgs: Tensor,
                num_subg: Tensor, num_node: Tensor, rwse: Tensor = None) -> Tensor:
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
        # print(self.idemb)
        subgs = subgs2sparse(subgs)
        B, Nm = x.shape[0], x.shape[1]
        Bidx = torch.arange(B, device=x.device)
        null_node_mask = torch.zeros((B, Nm+1), device=x.device)
        null_node_mask[Bidx, num_node] = 1
        null_node_mask = null_node_mask.cumsum_(dim=1) > 0
        null_node_mask = null_node_mask[:, :-1]

        subgbatch = self.num2batch(num_subg)  # 依次 ns 个 0, 1, ...

        x = self.inmlp(x)  # (B, Nm, hid_dim-se_dim)
        if self.se_dim > 0 and rwse is not None:
            rwse_emb = self.se_emb(rwse)  # (B, Nm, se_dim)
            x = torch.cat([x, rwse_emb], dim=-1)  # (B, Nm, hid_dim)
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

        if adj.dtype == torch.long:
            adj = self.edge_emb(adj)
        else:
            adj = adj.unsqueeze_(-1)  # subadj (B, Nm, Nm, d/1)
        for _ in range(self.num_layer):
            x = x + self.graph_mlps[_](torch.einsum("bijd,bjd->bid", adj, x))  # (B, Nm, d)
            x = F.dropout(x, self.drop_ratio, self.training)
        pre_x = x.sum(dim=1)
        x = x[subgbatch]  # (ns, Nm, d)
        adj_ = adj[subgbatch]  # (ns, Nm, Nm, d/1)

        ns = 0
        trn_idx = []
        for b in range(B):
            if num_subg[b] == 0:
                continue
            n = int(self.rate * num_subg[b]) + 1 if (self.training or self.ensemble_test) else num_subg[b]
            trn_idx.append(random.sample(range(ns, ns + num_subg[b]), n))
            ns += num_subg[b]
        trn_idx = list(_flatten(trn_idx))

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
        if self.final_concat == 'cat':
            x = torch.cat([x, pre_x], dim=-1)
        elif self.final_concat == 'add':
            x = x + pre_x
        elif self.final_concat == 'hadamard_product':
            x = x * pre_x
        return self.outmlp(x)

