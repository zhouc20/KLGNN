from turtle import forward
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, GATConv, global_max_pool


class AttenPool(nn.Module):
    def __init__(self, in_dim, out_dim, aggr_fn = TransformerConv, pool_fn=global_max_pool, **kwargs):
        super().__init__()
        self.aggr = aggr_fn(in_dim, out_dim, **kwargs)
        self.pool_fn = pool_fn


    def forward(self, x, subgbatch):
        '''
        x (\sum_i ns_i, d)
        subgbatch (\sum_i ns_i)
        '''
        ns = x.shape[0]
        ei = torch.arange(ns, device=x.device)
        ei = torch.stack((ei.unsqueeze(0).expand(ns, -1), ei.unsqueeze(1).expand(-1, ns)),dim=0).flatten(1, 2)
        mask = (subgbatch.unsqueeze(0)==subgbatch.unsqueeze(1)).flatten()
        ei = ei[:, mask]
        x = self.aggr(x, ei) # (\sum_i ns_i, d)
        x = self.pool_fn(x, subgbatch) # (batchsize, d)
        return x