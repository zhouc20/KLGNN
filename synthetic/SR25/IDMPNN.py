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
from math import factorial


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


class IDMPNN_full(nn.Module):
    '''
        This version runs message passing on the whole graph.
        Note that we need to differentiate SR graphs, so model parameters can be arbitrary and manually specified.
    '''

    num_layer: Final[int]

    def __init__(self,
                 k: int,
                 in_dim: int,
                 hid_dim: int,
                 out_dim,
                 num_layer: int,
                 num_layer_global: int = 0,
                 num_layer_id: int = 1,
                 num_layer_regression: int = 4,
                 max_nodez: int = None,
                 max_edgez: int = None,
                 node_pool: str = 'add',
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
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.SELU()) for _ in range(num_layer)
        ])
        self.id_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.SELU()) for _ in range(num_layer_id)
        ])
        self.global_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.SELU()) for _ in range(num_layer_global)
        ])
        self.setmlp1 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.SELU())
        self.setmlp2 = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.SELU())
        self.setmlp3 = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
                          nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),  # no layernorm
                          nn.ReLU(inplace=True)) for _ in range(num_layer_regression)
        ])
        self.outmlp = nn.Sequential(nn.Linear(hid_dim, out_dim))
        assert global_pool in ['max', 'add', 'mean', 'attention', 'mix']
        assert node_pool in ['max', 'add', 'mean', 'attention', 'mix']
        self.global_pool_type = global_pool
        self.global_pool = eval("global_" + global_pool + "_pool") if global_pool in ['max', 'add', 'mean'] else AttenPool(hid_dim, hid_dim)
        self.node_pool = eval("global_" + node_pool + "_pool") if node_pool in ['max', 'add', 'mean'] else AttenPool(hid_dim, hid_dim)
        self.inmlp = nn.Sequential(nn.Linear(1, hid_dim), nn.ReLU(inplace=True),
                                   nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim),  # no layernorm
                                   nn.ReLU(inplace=True))
        self.edge_emb_1 = nn.Linear(in_dim, hid_dim)
        self.edge_emb_2 = nn.Linear(in_dim, hid_dim)

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
                num_subg: Tensor) -> Tensor:
        '''
        Nm = max(Ni)
        x: (B, Nm, d)
        adj: (B, N, N)
        subadj: (ns, k, k)
        subgs: sparse(ns1+ns2+...+nsB, k)
        num_subg: (B) vector of ns
        num_node: (B) vector of N
        '''
        '''
        x (ns1+ns2+...+nsB, N_m, (k-1)!, d)
        '''

        subgs = subgs2sparse(subgs)
        if len(adj.shape) == 2:
            adj = adj.unsqueeze(0)
        N = adj.shape[-1]
        subgbatch = self.num2batch(num_subg)  # 依次 ns 个 0, 1, ...

        x = self.inmlp(x)  # (B, Nm, hid_dim)
        x = x[:, :N, :]  # (B, N, hid_dim)
        z = torch.ones(x.shape, requires_grad=True, device=x.device)
        z = z[subgbatch].unsqueeze(-2).repeat(
            1, 1, self.permdim, 1)  # (ns1+ns2+...+nsB, N, permdim, hid_dim)

        labelsubgptr, labelnodes, _ = subgs.csr()

        subgidx = torch.arange(labelnodes.shape[0], device=labelsubgptr.device)
        subgsize = torch.diff(labelsubgptr)
        cumdeg = torch.zeros_like(subgidx)
        cumdeg[labelsubgptr[1:-1]] = subgsize[:-1]

        cumdeg = cumdeg.cumsum_(dim=0)
        nodeidx = subgidx - cumdeg

        labelbatch = self.num2batch(subgsize)

        label = self.idemb(self.allperm[nodeidx])
        label = torch.autograd.Variable(label)
        ids = (torch.arange(self.k, device=x.device) + 1).repeat(z.shape[0])
        # * 100 to make forward passing easier to distinguish
        z[labelbatch, labelnodes] = torch.einsum("bpd,b->bpd", z[labelbatch, labelnodes] * label, ids * 1000) # (selectiondim, permdim, hid_dim)
        adj = adj.unsqueeze(-1)
        adj_ = adj[subgbatch]  # adj (ns, N, N, d/1)

        for _ in range(self.num_layer_id):
            z = z + self.id_mlps[_](noindexbmm(adj_, z))  # (ns, N, (k-1)!, d)
        z = self.setmlp1(z.sum(dim=2))  # (ns, N, d)
        x = z
        for _ in range(self.num_layer_global):
            x = x + self.global_mlps[_](torch.einsum("bijd,bjd->bjd", adj_, x))  # (ns, N, d)
        x = self.setmlp2(x.sum(dim=1))  # (ns, d)
        if self.global_pool_type == 'mix':
            x = self.global_pool(x, subgbatch) + global_max_pool(x, subgbatch)
        else:
            x = self.global_pool(x, subgbatch)
        for _ in range(self.num_layer_regression):
            x = x + self.setmlp3[_](x)
        return self.outmlp(x)
