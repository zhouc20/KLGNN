import random
from codecs import ascii_encode
from typing import Final, List, Tuple
from util import cumsum_pad0, deg2rowptr, extracttuple
import torch
import torch.nn as nn
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
        self.permdim = factorial(k - 1)
        allperm = extracttuple(torch.arange(k - 1) + 1, k - 1)
        allperm = torch.cat((torch.zeros(
            (allperm.shape[0], 1), dtype=allperm.dtype), allperm),
                            dim=-1)  # ((k-1)!, k)
        self.register_buffer("allperm", allperm.t())  #(k, (k-1)!)
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
        else:
            self.inmlp = nn.Sequential(nn.Linear(1, hid_dim), nn.ReLU(inplace=True),
                                       nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                                       nn.ReLU(inplace=True))
        self.edge_emb = nn.Linear(in_dim, hid_dim)
        # if max_edgez is not None:
        #     self.edge_emb = nn.Embedding(max_edgez + 1, hid_dim, padding_idx=0)

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
        x = torch.tensor(x, dtype=torch.float, device=x.device)
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
        subadj = subadj.unsqueeze_(-1)
        subadj = self.edge_emb(subadj)
        # else:
        #     subadj = subadj.unsqueeze_(-1)
        # subadj (B, Nm, Nm, d/1)
        # print(subadj.shape, x.shape)
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
        print(x)
        return self.outmlp(x)


class IDMPNN_Global_Local(nn.Module):
    num_layer: Final[int]

    def __init__(self,
                 k: int,
                 in_dim: int,
                 hid_dim: int,
                 out_dim,
                 num_layer: int,
                 num_layer_global: int = 0,
                 num_layer_local: int = 0,
                 max_nodez: int = None,
                 max_edgez: int = None,
                 global_pool: str = 'max'):
        super().__init__()

        self.k = k
        self.hid_dim = hid_dim
        self.idemb = nn.Embedding(k + 2, hid_dim)
        self.permdim = factorial(k)
        allperm = extracttuple(torch.arange(k) + 1, k)
        allperm = torch.cat((torch.zeros(
            (allperm.shape[0], 1), dtype=allperm.dtype), allperm),
                            dim=-1)  # ((k-1)!, k)
        self.register_buffer("allperm", allperm.t())  #(k, (k-1)!)
        self.num_layer = num_layer
        self.num_layer_global = num_layer_global
        self.num_layer_local = num_layer_local
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
        self.local_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_local)
        ])
        self.setmlp1 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp2 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp3 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.setmlp4 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     nn.ReLU(inplace=True))
        self.outmlp = nn.Linear(hid_dim, out_dim)
        assert global_pool in ['max', 'add', 'mean', 'attention']
        self.global_pool = eval("global_" + global_pool + "_pool") if global_pool != 'attention' else AttenPool(hid_dim, hid_dim)
        if max_nodez is not None:
            self.inmlp_mod = nn.Embedding(max_nodez + 1, hid_dim)
            self.inmlp = lambda x: self.inmlp_mod(x).squeeze(-2)
        else:
            self.inmlp = nn.Sequential(nn.Linear(1, hid_dim), nn.ReLU(inplace=True),
                                       nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                                       nn.ReLU(inplace=True))
        self.edge_emb = nn.Linear(in_dim, hid_dim)
        # if max_edgez is not None:
        #     self.edge_emb = nn.Embedding(max_edgez + 1, hid_dim, padding_idx=0)

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

    def forward(self, x: Tensor, adj: SparseTensor, subadj: SparseTensor, subgs: Tensor,
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
        x = torch.tensor(x, dtype=torch.float, device=x.device)
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
        z = x[labelbatch, labelnodes]
        z = z.reshape(x.shape[0], self.k, self.permdim, self.hid_dim)
        # print(z.shape)
        adj = adj.unsqueeze_(-1)
        adj = self.edge_emb(adj)
        subadj = subadj.unsqueeze_(-1)
        subadj = self.edge_emb(subadj)
        # else:
        #     subadj = subadj.unsqueeze_(-1)
        # subadj (B, Nm, Nm, d/1)
        # print(subadj.shape, x.shape)
        adj = adj[subgbatch]
        subadj = subadj[subgbatch]
        for _ in range(self.num_layer):
            x = x + self.mlps[_](noindexbmm(adj, x))  # (ns, Nm, k!, d)
        for _ in range(self.num_layer_local):
            z = z + self.local_mlps[_](noindexbmm(subadj, z))  # (ns, k, k!, d)
        x[null_node_mask[subgbatch]] = 0  # should not be here?
        # x = self.setmlp1(x.sum(dim=1))  # (ns, (k-1)!, d)
        # x = self.setmlp2(x.mean(dim=1))  # (ns, d)
        x = self.setmlp1(x.mean(dim=2))  # (ns, Nm, d)
        z = self.setmlp3(z.mean(dim=2))  # (ns, k, d)
        for _ in range(self.num_layer_global):
            x = x + self.global_mlps[_](torch.einsum("bijd,bjd->bjd", adj, x))  # (ns, Nm, d)
        x = self.setmlp2(x.sum(dim=1))  # (ns, d)
        z = self.setmlp4(z.sum(dim=1))  # (ns, d)
        # x = x + z
        x = z
        x = self.global_pool(x, subgbatch)
        # print(x)
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
                 max_edgez: int = None,
                 global_pool: str = 'max',
                 rate: float = 1.0):
        super().__init__()

        self.k = k
        self.hid_dim = hid_dim
        self.rate = rate
        self.idemb = nn.Embedding(k + 2, hid_dim)
        self.permdim = factorial(k - 1)
        allperm = extracttuple(torch.arange(k - 1) + 1, k - 1)
        allperm = torch.cat((torch.zeros(
            (allperm.shape[0], 1), dtype=allperm.dtype), allperm),
                            dim=-1)  # ((k-1)!, k)
        self.register_buffer("allperm", allperm.t())  #(k, (k-1)!)
        self.num_layer = num_layer
        self.num_layer_global = num_layer_global
        self.num_layer_id = num_layer_id
        self.num_layer_regression = num_layer_regression
        self.graph_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer)
        ])
        self.id_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_id)
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
        self.setmlp3 = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True)) for _ in range(num_layer_regression)
        ])
        self.outmlp = nn.Sequential(nn.Linear(hid_dim, out_dim))
        assert global_pool in ['max', 'add', 'mean', 'attention', 'mix']
        self.global_pool_type = global_pool
        self.global_pool = eval("global_" + global_pool + "_pool") if global_pool in ['max', 'add', 'mean'] else AttenPool(hid_dim, hid_dim)
        if max_nodez is not None:
            self.inmlp_mod = nn.Embedding(max_nodez + 1, hid_dim)
            self.inmlp = lambda x: self.inmlp_mod(x).squeeze(-2)
        else:
            self.inmlp = nn.Sequential(nn.Linear(1, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),
                          nn.ReLU(inplace=True))
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

    def forward(self, x: Tensor, adj: SparseTensor, subgs: Tensor,
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

        x = torch.tensor(x, dtype=torch.float, device=x.device)
        x = self.inmlp(x)  # (B, Nm, hid_dim)
        z = torch.ones(x.shape, requires_grad=True, device=x.device)
        z = z[subgbatch].unsqueeze(-2).repeat(
            1, 1, self.permdim, 1)  # (ns1+ns2+...+nsB, Nm, permdim, hid_dim)

        labelsubgptr, labelnodes, _ = subgs.csr()

        subgidx = torch.arange(labelnodes.shape[0], device=labelsubgptr.device)
        subgsize = torch.diff(labelsubgptr)
        cumdeg = torch.zeros_like(subgidx)
        cumdeg[labelsubgptr[1:-1]] = subgsize[:-1]

        cumdeg = cumdeg.cumsum_(dim=0)
        nodeidx = subgidx - cumdeg

        labelbatch = self.num2batch(subgsize)

        z[labelbatch, labelnodes] = z[labelbatch, labelnodes] * self.idemb(
            self.allperm[nodeidx])  # (selectiondim, permdim, hid_dim)
        # print(z.shape)
        # if subadj.dtype == torch.long:
        #     subadj = self.edge_emb(subadj)
        # else:
        adj = adj.unsqueeze_(-1)  # subadj (B, Nm, Nm, d/1)
        # print(subadj.shape)
        # print(x.shape)
        for _ in range(self.num_layer):
            x = x + self.graph_mlps[_](torch.einsum("bijd,bjd->bid", adj, x))  # (B, Nm, d)
        x = x[subgbatch]  # (ns, Nm, d)
        subadj_ = adj[subgbatch]  # subadj (ns, Nm, Nm, d/1)

        # if self.training:
        ns = 0
        trn_idx = []
        for b in range(B):
            trn_idx.append(random.sample(range(ns, ns + num_subg[b]), int(self.rate * num_subg[b]) + 1))
            ns += num_subg[b]
        # trn_idx = random.sample(range(ns), int(self.rate * ns))
        trn_idx = list(_flatten(trn_idx))

        # if not self.training:
        #     trn_idx = list(range(len(subgbatch)))
        z = z[trn_idx]
        subadj_ = subadj_[trn_idx]
        x = x[trn_idx]
        for _ in range(self.num_layer_id):
            z = z + self.id_mlps[_](noindexbmm(subadj_, z))  # (ns, Nm, (k-1)!, d)
        z[null_node_mask[subgbatch][trn_idx]] = 0  # should not be here?
        z = self.setmlp1(z.mean(dim=2))  # (ns, Nm, d)
        x = x + z  # 比例？乘法？
        for _ in range(self.num_layer_global):
            x = x + self.global_mlps[_](torch.einsum("bijd,bjd->bid", subadj_, x))  # (ns, Nm, d)
        x = self.setmlp2(x.sum(dim=1))  # (ns, d)
        if self.global_pool_type == 'mix':
            x = self.global_pool(x, subgbatch[trn_idx]) + global_max_pool(x, subgbatch[trn_idx])
        else:
            x = self.global_pool(x, subgbatch[trn_idx])
        for _ in range(self.num_layer_regression):
            x = x + self.setmlp3[_](x)  # (B, d)
        print(x)
        return self.outmlp(x)
