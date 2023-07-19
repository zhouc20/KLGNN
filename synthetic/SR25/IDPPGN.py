# PPGN layer
import torch 
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool
import torch_geometric.utils as pyg_utils
from torch import Tensor
from util import cumsum_pad0, deg2rowptr, extracttuple
from torch_sparse import SparseTensor
from tkinter import _flatten
import random
from math import factorial

BN = False


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


class DiscreteEncoder(nn.Module):
    def __init__(self, hidden_channels, max_num_features=10,
                 max_num_values=500):  # 10, change it for correctly counting number of parameters
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels)
                                         for i in range(max_num_features)])

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = 0
        for i in range(x.size(1)):
            out = out + self.embeddings[i](x[:, i])
        return out


class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=False, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i == 0 else n_hid,
                                               n_hid if i < nlayer - 1 else nout,
                                               bias=True if (
                                                                        i == nlayer - 1 and not with_final_activation and bias)  # TODO: revise later
                                                            or (not with_norm) else False)  # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.LayerNorm(n_hid if i < nlayer - 1 else nout) if with_norm else Identity()
                                    for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin == nout)

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer - 1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)

                # if self.residual:
        #     x = x + previous_x
        return x


class VNUpdate(nn.Module):
    def __init__(self, dim, with_norm=BN):
        """
        Intermediate update layer for the virtual node
        :param dim: Dimension of the latent node embeddings
        :param config: Python Dict with the configuration of the CRaWl network
        """
        super().__init__()
        self.mlp = MLP(dim, dim, with_norm=with_norm, with_final_activation=True, bias=not BN)

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, vn, x, batch):
        G = global_add_pool(x, batch)
        if vn is not None:
            G = G + vn
        vn = self.mlp(G)
        x = x + vn[batch]
        return vn, x


"""
   Original implementation of PPGN
"""

class PPGN(nn.Module):
    def __init__(self, nin, nhid, nout, nlayer, depth_of_mlp=2):
        super().__init__()
        # First part - sequential mlp blocks
        self.reg_blocks = nn.ModuleList([RegularBlock(nhid, nhid, depth_of_mlp) for i in range(nlayer)])
        # Second part
        # self.norm = Identity() # 
        self.norm = nn.BatchNorm1d(2*nhid)
        self.output_encoder = MLP(2*nhid, nout, nlayer=2, with_final_activation=False)
        self.edge_emb = nn.Embedding(2, nhid, padding_idx=0)
        self.inmlp = nn.Sequential(nn.Linear(nin, nhid), nn.ReLU(inplace=True),
                                   nn.Linear(nhid, nhid), nn.LayerNorm(nhid),  # no layernorm
                                   nn.ReLU(inplace=True))

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.output_encoder.reset_parameters()
        for reg in self.reg_blocks:
            reg.reset_parameters()

    def forward_original(self, x, edge_index, edge_attr, batch):
        # to dense first
        x, adj, mask_x = to_dense_batch(x, edge_index, edge_attr, batch) # B x N_max x N_max x F

        ### TODO: for PPGN-AK we need to make N_max smaller, by make batch hasing more disconnected component

        # combine x and adj 
        idx_tmp = range(x.size(1))
        adj[:, idx_tmp, idx_tmp, :] = x
        x = torch.transpose(adj, 1, 3) # B x F x N_max x N_max 

        # create new mask 
        mask_adj = mask_x.unsqueeze(2) * mask_x.unsqueeze(1) # Bx N_max x N_max

        for block in self.reg_blocks:
            # consider add residual connection here?
            x = block(x, mask_adj)

        # 2nd order to 1st order matrix
        diag_x = x[:, :, idx_tmp, idx_tmp] # B x F x N_max
        offdiag_x = x.sum(dim=-1) - diag_x # B x F x N_max,  use summation here, can change to mean or max. 
        x = torch.cat([diag_x, offdiag_x], dim=1)
        x = self.norm(x).transpose(1, 2) # B x N_max x 2F

        # to sparse x 
        x = x.reshape(-1, x.size(-1))[mask_x.reshape(-1)] # BN x F

        # transform feature by mlp
        x = self.output_encoder(x) # BN x F

        return x

    def forward(self, x, adj):
        # to dense first
        # x, adj, mask_x = to_dense_batch(x, edge_index, edge_attr, batch) # B x N_max x N_max x F
        B, Nm = x.shape[0], x.shape[1]
        Nm = adj.shape[-1]
        x = x[:, :Nm, :] * 20
        num_node = torch.tensor((Nm), device=x.device, dtype=torch.long)
        Bidx = torch.arange(B, device=x.device)
        null_node_mask = torch.zeros((B, Nm + 1), device=x.device)
        null_node_mask[Bidx, num_node] = 1
        null_node_mask = null_node_mask.cumsum_(dim=1) > 0
        null_node_mask = null_node_mask[:, :-1]

        ### TODO: for PPGN-AK we need to make N_max smaller, by make batch hasing more disconnected component

        # combine x and adj
        x = self.inmlp(x)
        idx_tmp = range(x.size(1))
        # adj = adj.unsqueeze(-1) * 20
        adj = self.edge_emb(adj.to(torch.long))
        adj[:, idx_tmp, idx_tmp, :] = x
        x = torch.transpose(adj, 1, 3) # B x F x N_max x N_max

        # create new mask
        mask_adj = ~null_node_mask.unsqueeze(2) * ~null_node_mask.unsqueeze(1) # Bx N_max x N_max

        for block in self.reg_blocks:
            # consider add residual connection here?
            x = block(x, mask_adj)

        # 2nd order to 1st order matrix
        diag_x = x[:, :, idx_tmp, idx_tmp] # B x F x N_max
        offdiag_x = x.sum(dim=-1) - diag_x # B x F x N_max,  use summation here, can change to mean or max.
        x = torch.cat([diag_x, offdiag_x], dim=1)
        x = self.norm(x).transpose(1, 2) # B x N_max x 2F

        # to sparse x
        x = x.reshape(-1, x.size(-1))[~null_node_mask.reshape(-1)] # BN x F

        # transform feature by mlp
        x = self.output_encoder(x) # BN x F
        x = torch.sum(x, dim=0)

        return x


def subgs2sparse(subgs: Tensor) -> SparseTensor:
    mask = (subgs >= 0)
    deg = torch.sum(mask, dim=1)
    rowptr = deg2rowptr(deg)
    col = subgs.flatten()[mask.flatten()]
    return SparseTensor(rowptr=rowptr, col=col).device_as(mask)


"""
    Implementation of IDPPGN. k IDs corresponds to 2,k-FWL.
"""

class IDPPGN(nn.Module):
    def __init__(self, k, nin, nhid, nout, nlayer, depth_of_mlp=2, rate=0.1, cat='add', max_edgez=None):
        super().__init__()
        self.k = k
        allperm = extracttuple(torch.arange(k), k)
        self.register_buffer("allperm", allperm.t())
        self.idemb = nn.Embedding(k + 2, nhid)
        self.permdim = factorial(k)
        self.inmlp = nn.Sequential(nn.Linear(nin, nhid), nn.ReLU(inplace=True),
                          nn.Linear(nhid, nhid), nn.LayerNorm(nhid), # no layernorm
                          nn.ReLU(inplace=True))
        # First part - sequential mlp blocks
        self.reg_blocks = nn.ModuleList([RegularBlock(nhid, nhid, depth_of_mlp) for _ in range(nlayer)])
        # Second part
        # self.norm = Identity() #
        self.norm = nn.BatchNorm1d(2*nhid)
        self.output_encoder = MLP(2*nhid, nout, nlayer=2, with_final_activation=False)
        self.rate = rate
        assert cat in ['add', 'hadamard_product', 'cat', 'none']
        self.cat = cat
        self.nhid = nhid
        self.edge_emb = nn.Embedding(2, nhid, padding_idx=0)
        # self.setmlp1 = nn.Sequential(nn.SELU())
        # self.setmlp2 = nn.Sequential(nn.SELU())
        # self.setmlp3 = nn.Sequential(nn.SELU())
        self.setmlp1 = nn.Sequential(nn.Linear(2*nhid, 2*nhid), nn.SELU())
        self.setmlp2 = nn.Sequential(nn.Linear(2*nhid, 2*nhid), nn.SELU())
        self.setmlp3 = nn.Sequential(nn.Linear(2*nhid, 2*nhid), nn.SELU())

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.output_encoder.reset_parameters()
        for reg in self.reg_blocks:
            reg.reset_parameters()

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

    def forward(self, x, adj: Tensor, subgs: Tensor,
                num_subg: Tensor, num_node: Tensor):

        # x = x.unsqueeze_(0)  # bs = 1
        # num_subg = torch.tensor([1], device=x.device)
        # subgs = subgs[0].unsqueeze(0)
        subgs = subgs2sparse(subgs)
        B, Nm = x.shape[0], x.shape[1]
        Nm = adj.shape[-1]
        x = x[:, :Nm, :]
        # print(Nm)
        num_node = torch.tensor((Nm), device=x.device, dtype=torch.long)
        Bidx = torch.arange(B, device=x.device)
        null_node_mask = torch.zeros((B, Nm + 1), device=x.device)
        null_node_mask[Bidx, num_node] = 1
        null_node_mask = null_node_mask.cumsum_(dim=1) > 0
        null_node_mask = null_node_mask[:, :-1]
        subgbatch = self.num2batch(num_subg)
        labelsubgptr, labelnodes, _ = subgs.csr()

        subgidx = torch.arange(labelnodes.shape[0], device=labelsubgptr.device)
        subgsize = torch.diff(labelsubgptr)
        cumdeg = torch.zeros_like(subgidx)
        cumdeg[labelsubgptr[1:-1]] = subgsize[:-1]

        cumdeg = cumdeg.cumsum_(dim=0)
        nodeidx = subgidx - cumdeg

        labelbatch = self.num2batch(subgsize)

        adj = adj.unsqueeze(-1).repeat(1, 1, 1, self.nhid) * 1.
        x = self.inmlp(x) * 1.
        x = x[subgbatch].unsqueeze(-2).repeat(1, 1, self.permdim, 1)  # (ns, Nm, permdim, d)

        adj = adj[subgbatch]
        label = self.idemb(self.allperm[nodeidx])
        label = torch.autograd.Variable(label) * 10.

        ids = (torch.arange(self.k, device=x.device) + 1).repeat(x.shape[0])
        if self.cat == 'add':
            x[labelbatch, labelnodes] = torch.einsum("bpd,b->bpd", x[labelbatch, labelnodes] + label, ids * 20.)
        elif self.cat == 'hadamard_product':
            x[labelbatch, labelnodes] = torch.einsum("bpd,b->bpd", x[labelbatch, labelnodes] * label, ids * 20.)

        x = x.permute(0, 2, 1, 3).reshape(-1, Nm, self.nhid)  # (ns * #perm, Nm, hid_dim)
        adj = adj.unsqueeze(1).repeat(1, self.permdim, 1, 1, 1).reshape(x.shape[0], Nm, Nm, self.nhid)

        # combine x and adj
        idx_tmp = range(Nm)
        adj[:, idx_tmp, idx_tmp, :] = x
        x = torch.transpose(adj, 1, 3) # ns * #perm x F x N_max x N_max

        # create new mask
        mask_adj = (~null_node_mask).unsqueeze(2) * (~null_node_mask).unsqueeze(1) # Bx N_max x N_max
        mask_adj = mask_adj[subgbatch]
        mask_adj = mask_adj.unsqueeze(1).repeat(1, self.permdim, 1, 1).reshape(x.shape[0], Nm, Nm)

        for block in self.reg_blocks:
            # consider add residual connection here?
            x = block(x, mask_adj)

        # 2nd order to 1st order matrix
        diag_x = x[:, :, idx_tmp, idx_tmp] # ns * #perm x F x N_max
        offdiag_x = x.sum(dim=-1) - diag_x # ns * #perm x F x N_max,  use summation here, can change to mean or max.
        x = torch.cat([diag_x, offdiag_x], dim=1)
        x = x.transpose(1, 2) # ns * #perm x N_max x 2F

        # 2 poolings
        pool_batch = subgbatch.unsqueeze(1).repeat(1, self.permdim).reshape(-1)
        x = global_add_pool(x.reshape(-1, Nm * 2 * self.nhid), pool_batch)
        x = self.setmlp1(x.reshape(B * Nm, 2 * self.nhid))  # [mask_x.reshape(-1)] # BN x 2F
        subgnode_batch = torch.arange(B, device=x.device).unsqueeze(1).repeat(1, Nm).reshape(-1)
        x = self.setmlp2(global_add_pool(x, subgnode_batch))

        # transform feature by mlp
        x = self.output_encoder(x)  # B x output

        return x


#######################################################################################################
# Helpers for PPGN, from original repo: https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch

class RegularBlock(nn.Module):
    """
    Imputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes, multiply the result, and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth
    :return: Tensor of shape N x output_depth x m x m
    """
    def __init__(self, in_features, out_features, depth_of_mlp=2):
        super().__init__()
        self.mlp1 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.mlp2 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.skip = SkipConnection(in_features+out_features, out_features)
    
    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
        self.skip.reset_parameters()

    def forward(self, inputs, mask):
        mask = mask.unsqueeze(1).to(inputs.dtype)
        mlp1 = self.mlp1(inputs) * mask
        mlp2 = self.mlp2(inputs) * mask

        mult = torch.matmul(mlp1, mlp2)

        out = self.skip(in1=inputs, in2=mult) * mask
        return out


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn=nn.functional.relu_):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for _ in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            in_features = out_features

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs:
            out = self.activation(conv_layer(out))

        return out

class SkipConnection(nn.Module):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param in_features: d1+d2
    :param out_features: output num of features
    :return: Tensor of shape N x output_depth x m x m
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, in1, in2):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out


def to_dense_batch(x, edge_index, edge_attr, batch, max_num_nodes=None):
    x, mask = pyg_utils.to_dense_batch(x, batch, max_num_nodes=max_num_nodes) # B x N_max x F
    adj = pyg_utils.to_dense_adj(edge_index, batch, edge_attr, max_num_nodes=max_num_nodes)
    # x:  B x N_max x F
    # mask: B x N_max
    # adj: B x N_max x N_max x F
    return x, adj, mask
#
#######################################################################################################