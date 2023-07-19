import torch
from torch import Tensor
from typing import List
from typing import List
import torch
from torch_geometric.data import Data as PygData
from torch import Tensor


def cumsum_pad0(num: Tensor):
    ret = torch.empty_like(num)
    ret[0] = 0
    ret[1:] = torch.cumsum(num[:-1], dim=0)
    return ret


def deg2rowptr(num: Tensor):
    ret = torch.empty((num.shape[0]+1), device=num.device, dtype=num.dtype)
    ret[0] = 0
    ret[1:] = torch.cumsum(num, dim=0)
    return ret

def extracttuple(elems: Tensor, k: int) -> Tensor:
    '''
    extract k-tuple from set
    elems : (N)
    '''
    def removedegeneratedtuple(tuples: Tensor):
        '''
        tuples (N, k)
        '''
        sorted_tuple = torch.sort(tuples, dim=-1)[0]
        mask = (torch.diff(sorted_tuple, dim=-1) > 0).all(dim=-1)
        return tuples[mask]

    N = elems.shape[0]
    elemslist: List[Tensor] = []
    for i in range(k):
        shapevec = [1] * (k)
        shapevec[i] = -1
        elemslist.append(elems.reshape(shapevec))
        shapevec = [N] * k
        elemslist[-1] = elemslist[-1].expand(shapevec)
    tuples = torch.stack(elemslist, dim=-1).reshape(-1, k)
    return removedegeneratedtuple(tuples)


def extractsubset(elems: Tensor, k: int) -> Tensor:
    '''
    extract k-tuple from set
    elems : (N)
    '''
    N = elems.shape[0]
    elemslist: List[Tensor] = []
    for i in range(k):
        shapevec = [1] * (k)
        shapevec[i] = -1
        elemslist.append(elems.reshape(shapevec))
        shapevec = [N] * k
        elemslist[-1] = elemslist[-1].expand(shapevec)
    tuples = torch.stack(elemslist, dim=-1).reshape(-1, k)
    mask = (torch.diff(tuples, dim=-1) > 0).all(dim=-1)
    return tuples[mask]