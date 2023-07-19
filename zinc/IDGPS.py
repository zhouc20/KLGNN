import random
from codecs import ascii_encode
from typing import Final, List, Tuple, Optional
from util import cumsum_pad0, deg2rowptr, extracttuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool
from subgraphpooling import AttenPool
from torch_sparse import SparseTensor
from tkinter import _flatten
import math

# import torch
# from fairseq import utils
# from fairseq.modules.fairseq_dropout import FairseqDropout
# from fairseq.modules.quant_noise import quant_noise
# from torch import Tensor, nn


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
                 max_edgez: int = None,
                 node_pool: str = 'mean',
                 subgraph_pool: str = 'add',
                 global_pool: str = 'max',
                 rate: float = 1.0,
                 cat: str = 'add',
                 drop_ratio: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_perm: float = 1.0,
                 norm_type: str = 'layer',
                 ensemble_test: bool = False,
                 final_concat: str = 'none',
                 num_head: int = 8,
                 local_MPNN: bool = False,
                 central_encoding: bool = False,
                 attn_bias: bool = False,
                 rw_step: int = 20,
                 se_dim: int = 16,
                 se_type: str = 'linear'):
        super().__init__()

        self.k = k
        self.hid_dim = hid_dim
        self.se_dim = se_dim
        self.rate = rate
        self.ensemble_test = ensemble_test
        self.drop_ratio = drop_ratio
        self.attn_drop = attn_drop
        self.idemb = nn.Embedding(k + 2, hid_dim)
        self.permdim = factorial(k)
        self.drop_permdim = int(self.permdim * drop_perm)
        allperm = extracttuple(torch.arange(k), k)
        self.register_buffer("allperm", allperm.t())  # (k, k!)
        self.num_layer = num_layer
        self.num_layer_global = num_layer_global
        self.num_layer_id = num_layer_id
        self.num_layer_regression = num_layer_regression
        self.central_encoding = central_encoding
        self.attn_bias = attn_bias
        self.local_MPNN = local_MPNN
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
        self.graph_self_attn = nn.ModuleList([MultiheadAttention(
            self.hid_dim, num_head, dropout=drop_ratio, batch_first=True) for _ in range(num_layer)])
        self.global_self_attn = nn.ModuleList([MultiheadAttention(
            self.hid_dim, num_head, dropout=drop_ratio, batch_first=True) for _ in range(num_layer_global)])
        # alternatively, use standard MultiheadAttention
        # self.graph_self_attn = nn.ModuleList([torch.nn.MultiheadAttention(
        #     self.hid_dim, num_head, dropout=drop_ratio, batch_first=True) for _ in range(num_layer)])
        # self.global_self_attn = nn.ModuleList([torch.nn.MultiheadAttention(
        #     self.hid_dim, num_head, dropout=drop_ratio, batch_first=True) for _ in range(num_layer_global)])
        self.graph_attn_bias = nn.ModuleList([AttentionBias(num_head) for _ in range(num_layer)])
        self.global_attn_bias = nn.ModuleList([AttentionBias(num_head) for _ in range(num_layer_global)])
        self.graph_central_encode = nn.ModuleList([nn.Embedding(10, hid_dim) for _ in range(num_layer)])
        self.global_central_encode = nn.ModuleList([nn.Embedding(10, hid_dim) for _ in range(num_layer_global)])
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
        if max_nodez is not None:
            self.inmlp_mod = nn.Embedding(max_nodez + 1, hid_dim - se_dim)
            self.inmlp = lambda x: self.inmlp_mod(x).squeeze(-2)
        if max_edgez is not None:
            self.edge_emb = nn.Embedding(max_edgez + 1, hid_dim, padding_idx=0)
        assert se_type in ['linear', 'mlp']
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

    def forward(self, x: Tensor, degree: Tensor, rwse: Tensor, adj: SparseTensor, distance: Tensor, subgs: Tensor,
                num_subg: Tensor, num_node: Tensor) -> Tensor:
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
        # print(degree.shape)
        # print(distance.shape)
        subgs = subgs2sparse(subgs)
        B, Nm = x.shape[0], x.shape[1]
        Bidx = torch.arange(B, device=x.device)
        null_node_mask = torch.zeros((B, Nm+1), device=x.device)
        null_node_mask[Bidx, num_node] = 1
        null_node_mask = null_node_mask.cumsum_(dim=1) > 0
        null_node_mask = null_node_mask[:, :-1]

        subgbatch = self.num2batch(num_subg)  # 依次 ns 个 0, 1, ...

        x = self.inmlp(x)  # (B, Nm, hid_dim - se_dim)
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

        adj_original = adj
        if adj.dtype == torch.long:
            adj = self.edge_emb(adj)
        else:
            adj = adj.unsqueeze_(-1)  # subadj (B, Nm, Nm, d/1)
        for _ in range(self.num_layer):
            attn_bias = self.graph_attn_bias[_](distance, adj_original) if self.attn_bias else None
            x2 = x + self.graph_central_encode[_](degree) if self.central_encoding else x
            x2 = self.graph_self_attn[_](x2, x2, x2, attn_bias,
                               attn_mask=None,
                               key_padding_mask=null_node_mask,
                               need_weights=False)[0]
            x2 = self.graph_attn_norm[_](x2)
            x2 = F.dropout(x2, self.attn_drop, self.training)
            if self.local_MPNN:
                x1 = self.graph_mlps[_](torch.einsum("bijd,bjd->bid", adj, x))  # (B, Nm, d)
                x1 = F.dropout(x1, self.drop_ratio, self.training)
                x = x + x1 + x2
            else:
                x = x + x2
            x = x + self.graph_ffn[_](x)
            x = F.dropout(x, self.drop_ratio, self.training)
        pre_x = x.sum(dim=1)
        #x = x[subgbatch]  # (ns, Nm, d)
        adj_ = adj[subgbatch]  # subadj (ns, Nm, Nm, d/1)

        # if self.training:
        if self.num_layer_id > 0:
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
            x2 = x + self.global_central_encode[_](degree) if self.central_encoding else x
            attn_bias = self.global_attn_bias[_](distance, adj_original) if self.attn_bias else None
            x2 = self.global_self_attn[_](x2, x2, x2, attn_bias,
                                         attn_mask=None,
                                         key_padding_mask=null_node_mask,
                                         need_weights=False)[0]
            # self.attn_weights = A.detach().cpu()
            x2 = self.global_attn_norm[_](x2)
            x2 = F.dropout(x2, self.attn_drop, self.training)
            if self.local_MPNN:
                x1 = self.global_mlps[_](torch.einsum("bijd,bjd->bid", adj, x))  # (B, Nm, d)
                x1 = F.dropout(x1, self.drop_ratio, self.training)
                x = x + x1 + x2
            else:
                x = x + x2
            x = x + self.global_ffn[_](x)
            x = F.dropout(x, self.drop_ratio, self.training)
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


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=True,
        q_noise=0.0,
        qn_block_size=8,
        batch_first: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        # self.dropout_module = FairseqDropout(
        #     dropout, module_name=self.__class__.__name__
        # )
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.batch_first = batch_first

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        # self.k_proj = quant_noise(
        #     nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        # )
        # self.v_proj = quant_noise(
        #     nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        # )
        # self.q_proj = quant_noise(
        #     nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        # )
        #
        # self.out_proj = quant_noise(
        #     nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        # )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        if self.batch_first:
            query = query.transpose(1, 0)
            key = key.transpose(1, 0)
            value = value.transpose(1, 0)

        tgt_len, bsz, embed_dim = query.size()
        # bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.reshape(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = torch.softmax(
            attn_weights, dim=-1 #, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights, self.dropout)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        if self.batch_first:
            attn = attn.permute(1, 0, 2)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


class AttentionBias(nn.Module):
    def __init__(self,
                 n_head: int,
                 # max_degree: int = 10,
                 max_distance: int = 37,
                 max_bond_type: int = 4):
        super().__init__()

        self.n_head = n_head
        # self.max_degree = max_degree
        self.max_distance = max_distance
        self.max_bond_type = max_bond_type

        self.edge_encode = nn.Embedding(self.max_bond_type, self.n_head, padding_idx=0)
        self.distance_encode = nn.Embedding(self.max_distance, self.n_head)

    def forward(self, distance, adj):
        x1 = self.edge_encode(adj).permute(0, 3, 1, 2)
        x2 = self.distance_encode(distance).permute(0, 3, 1, 2)
        x = x1 + x2
        return x

