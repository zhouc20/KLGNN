import time
import numpy as np
import torch.nn as nn
import torch
# from IDMPNN import IDMPNN
from IDMPNN_Global import IDMPNN_Global_new, IDMPNN_Global_parallel, ID_GINE, IDMPNN_Discrete, IDMPNN_Transformer
from preprocess import graph2IDsubgraph, graph2IDsubgraph_global, graph2IDsubgraph_global_new, graph2IDsubgraph_cluster
from datalist import DataListSet
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch_geometric.data import DataLoader as PygDataloader
from torch_sparse import SparseTensor
import argparse
import random
import os, sys
import math
import errno
import matplotlib.pyplot as plt
from dataset import load_dataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="star")
parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
parser.add_argument("--model", type=str, default='GINE')
parser.add_argument("--k", type=int, default=4)
parser.add_argument("--no_cluster", action='store_true', default=False)
parser.add_argument("--resolution", type=float, default=0.5, help='resolution of multi-level clustering')
parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--strategy', type=str, default='bfs')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--drop_ratio', type=float, default=0.0, help='drop out rate')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cos_lr', action='store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--loss', type=str, default='L1Loss')
parser.add_argument('--in_dim', type=int, default=9)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--out_dim', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_layers_id', type=int, default=0)
parser.add_argument('--num_layers_global', type=int, default=1)
parser.add_argument('--num_layers_regression', type=int, default=1)
parser.add_argument('--norm_type', type=str, default='layer')
parser.add_argument('--cat', type=str, default='add')
parser.add_argument('--node_pool', type=str, default='mean')
parser.add_argument('--subgraph_pool', type=str, default='add')
parser.add_argument('--global_pool', type=str, default='max')
parser.add_argument('--full_graph', action='store_true', default=False)
parser.add_argument('--rate', type=float, default=1.0)
parser.add_argument('--drop_perm', type=float, default=1.0, help='proportion of number of permutations')
parser.add_argument('--mask_value', type=float, default=1.0)
parser.add_argument('--num_tasks', type=int, default=1)
parser.add_argument('--no', type=int, default=0)
parser.add_argument('--name', type=str, default='')
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--aggregate', action='store_true', default=False)
parser.add_argument('--bi_aggregate', action='store_true', default=False)
parser.add_argument('--ak', action='store_true', default=False)
parser.add_argument('--ensemble_train', action='store_true', default=False)
parser.add_argument('--ensemble_test', action='store_true', default=False)
parser.add_argument('--sample_times', type=int, default=1)
parser.add_argument('--ensemble_mode', type=str, default='mean')
parser.add_argument('--distlimit', type=int, default=-1)
parser.add_argument('--distlimitu', type=int, default=3)
parser.add_argument('--distlimitl', type=int, default=-1)
parser.add_argument('--factor', type=float, default=0.5)  # 0.5
parser.add_argument('--patience', type=int, default=20)  # 3
parser.add_argument('--num_head', type=int, default=8)

args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, cos_lr, training_configurations):
    """Sets the learning rate"""

    if not cos_lr:
        if epoch in training_configurations['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations['lr_decay_rate']

    else:
        warm_up = 50
        if epoch < warm_up:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * training_configurations['initial_learning_rate']\
                                    * (1 + math.cos(math.pi * epoch / training_configurations['epochs'])) * epoch / warm_up \
                                    + (1 - epoch / warm_up) * training_configurations['initial_learning_rate'] * 0.1
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * training_configurations['initial_learning_rate']\
                                    * (1 + math.cos(math.pi * epoch / training_configurations['epochs']))


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def pre_transform_cluster(g):
    return graph2IDsubgraph_cluster(g, args.k, 121, args.distlimitl, args.distlimitu, args.resolution)

def pre_transform(g):
    return graph2IDsubgraph_global_new(g, args.k, 121, args.distlimitl, args.distlimitu)


k = args.k
depth = args.depth
strategy = args.strategy
assert strategy in ['neighbor', 'path', 'subgraph', 'bfs']
device = torch.device("cuda")

# glist, trn_idx, val_idx, tst_idx = load_dataset(args.dataset, args.task)
print('load dataset!', flush=True)

t1 = time.time()
processed_path = f"data/{args.dataset}_{k}_{args.distlimitl}_{args.distlimitu}_res_{args.resolution}.pt" if not args.no_cluster else f"data/{args.dataset}_{k}_{args.distlimitl}_{args.distlimitu}.pt"
if os.path.exists(processed_path):
    dataset = torch.load(processed_path, map_location="cpu")
elif not args.no_cluster:
    # datalist = [
    #     graph2IDsubgraph_cluster(dat, k, max([g.x.shape[0] for g in glist]),
    #                                 args.distlimitl, args.distlimitu, args.resolution)
    #     for dat in glist
    # ]
    # dataset = DataListSet(datalist)
    dataset = PygGraphPropPredDataset(args.dataset, 'data', pre_transform=pre_transform_cluster)
    torch.save(dataset, processed_path)
else:
    dataset = PygGraphPropPredDataset(args.dataset, 'data', pre_transform=pre_transform)
    torch.save(dataset, processed_path)
dataset.data = dataset.data.to(device)
split_idx = dataset.get_idx_split()
train_dataset, val_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
train_dataset_1 = []
train_dataset_2 = []
train_dataset_3 = []
train_dataset_4 = []
train_dataset_5 = []
# print(len(train_dataset))
# print(train_dataset[0])
for i in range(len(train_dataset)):
    if train_dataset[i].Nm == 30 and train_dataset[i].num_node > 6:
        train_dataset_1.append(i)
    if train_dataset[i].Nm == 50:
        train_dataset_2.append(i)
    if train_dataset[i].Nm == 80:
        train_dataset_3.append(i)
    if train_dataset[i].Nm == 100:
        train_dataset_4.append(i)
    if train_dataset[i].Nm == 222:
        train_dataset_5.append(i)
train_dataset_1 = train_dataset[train_dataset_1]
train_dataset_2 = train_dataset[train_dataset_2]
train_dataset_3 = train_dataset[train_dataset_3]
train_dataset_4 = train_dataset[train_dataset_4]
train_dataset_5 = train_dataset[train_dataset_5]
# print(len(train_dataset_1), len(train_dataset_2), len(train_dataset_3), len(train_dataset_4), len(train_dataset_5))
val_dataset_1 = []
val_dataset_2 = []
val_dataset_3 = []
val_dataset_4 = []
val_dataset_5 = []
for i in range(len(val_dataset)):
    if val_dataset[i].Nm == 30 and val_dataset[i].num_node > 6:
        val_dataset_1.append(i)
    if val_dataset[i].Nm == 50:
        val_dataset_2.append(i)
    if val_dataset[i].Nm == 80:
        val_dataset_3.append(i)
    if val_dataset[i].Nm == 100:
        val_dataset_4.append(i)
    if val_dataset[i].Nm == 222:
        val_dataset_5.append(i)
val_dataset_1 = val_dataset[val_dataset_1]
val_dataset_2 = val_dataset[val_dataset_2]
val_dataset_3 = val_dataset[val_dataset_3]
val_dataset_4 = val_dataset[val_dataset_4]
val_dataset_5 = val_dataset[val_dataset_5]
test_dataset_1 = []
test_dataset_2 = []
test_dataset_3 = []
test_dataset_4 = []
test_dataset_5 = []
for i in range(len(test_dataset)):
    if test_dataset[i].Nm == 30 and test_dataset[i].num_node > 6:
        test_dataset_1.append(i)
    if test_dataset[i].Nm == 50:
        test_dataset_2.append(i)
    if test_dataset[i].Nm == 80:
        test_dataset_3.append(i)
    if test_dataset[i].Nm == 100:
        test_dataset_4.append(i)
    if test_dataset[i].Nm == 222:
        test_dataset_5.append(i)
test_dataset_1 = test_dataset[test_dataset_1]
test_dataset_2 = test_dataset[test_dataset_2]
test_dataset_3 = test_dataset[test_dataset_3]
test_dataset_4 = test_dataset[test_dataset_4]
test_dataset_5 = test_dataset[test_dataset_5]
# for dset in [train_dataset_1, train_dataset_2, train_dataset_4, train_dataset_4, train_dataset_5, val_dataset_1, val_dataset_2, val_dataset_4, val_dataset_4, val_dataset_5, test_dataset_1, test_dataset_2, test_dataset_4, test_dataset_4, test_dataset_5]:
#     # for i in range(len(dset)):
#     #     Nm = dset[i].Nm
#     #     dset[i].x = torch.cat((dset[i].x, torch.zeros((Nm - dset[i].x.shape[0], 9),
#     #                                    dtype=dset[i].x.dtype, device=device))).unsqueeze_(0)
#     #     adj = SparseTensor(row=dset[i].edge_index[0],
#     #                        col=dset[i].edge_index[1],
#     #                        value=dset[i].edge_attr,
#     #                        sparse_sizes=(Nm, Nm)).coalesce()
#     #     dset[i].adj = adj.to_dense().unsqueeze_(0).to(torch.long)
#     for data in dset:
#         Nm = data.Nm
#         data.x = torch.cat((data.x, torch.zeros((Nm - data.x.shape[0], 9),
#                                        dtype=data.x.dtype, device=device))).unsqueeze_(0)
#         adj = SparseTensor(row=data.edge_index[0],
#                            col=data.edge_index[1],
#                            value=data.edge_attr,
#                            sparse_sizes=(Nm, Nm)).coalesce()
#         data.adj = adj.to_dense().unsqueeze_(0).to(torch.long)
# print(train_dataset[0])

train_curve = []
valid_curve = []
test_curve = []
best_val_score = float("inf")
bs = args.batch_size
train_dataloader_1 = PygDataloader(train_dataset_1, batch_size=bs, shuffle=True)
train_dataloader_2 = PygDataloader(train_dataset_2, batch_size=bs, shuffle=True)
train_dataloader_3 = PygDataloader(train_dataset_3, batch_size=int(bs/2), shuffle=True)
train_dataloader_4 = PygDataloader(train_dataset_4, batch_size=int(bs/4), shuffle=True)
train_dataloader_5 = PygDataloader(train_dataset_5, batch_size=1, shuffle=True)
val_dataloader_1 = PygDataloader(val_dataset_1, batch_size=bs)
val_dataloader_2 = PygDataloader(val_dataset_2, batch_size=bs)
val_dataloader_3 = PygDataloader(val_dataset_3, batch_size=int(bs/2))
val_dataloader_4 = PygDataloader(val_dataset_4, batch_size=int(bs/4))
val_dataloader_5 = PygDataloader(val_dataset_5, batch_size=1)
test_dataloader_1 = PygDataloader(test_dataset_1, batch_size=bs)
test_dataloader_2 = PygDataloader(test_dataset_2, batch_size=bs)
test_dataloader_3 = PygDataloader(test_dataset_3, batch_size=int(bs/2))
test_dataloader_4 = PygDataloader(test_dataset_4, batch_size=int(bs/4))
test_dataloader_5 = PygDataloader(test_dataset_5, batch_size=1)

evaluator = Evaluator(args.dataset)
print(evaluator.name)
# datalist = [dat for dat in dataset]
# Nm = 0
# N30 = 0
# N50 = 0
# N80 = 0
# N100 = 0
# N222 = 2
# for i in range(len(datalist)):
#     if datalist[i].x.shape[0]>Nm:
#         Nm = datalist[i].x.shape[0]
#     if datalist[i].x.shape[0] < 30:
#         N30 += 1
#     elif datalist[i].x.shape[0] < 50:
#         N50 += 1
#     elif datalist[i].x.shape[0] < 80:
#         N80 += 1
#     elif datalist[i].x.shape[0] < 100:
#         N100 += 1
#     else:
#         N222 += 1
# print(Nm)
# print(N30, N50, N80, N100, N222)
# processed_path = f"data/{args.dataset}_{k}_{args.distlimit}.pt"
# if os.path.exists(processed_path):
#     dataset = torch.load(processed_path, map_location="cpu")
# else:
#     datalist = [graph2IDsubgraph_global(dat, k, max([g.x.shape[0] for g in glist]), args.distlimit) for dat in glist]
#     dataset = DataListSet(datalist)
#     torch.save(dataset, processed_path)

print(f"preprocess {int(time.time()-t1)} s", flush=True)
print(len(dataset))

loss_fn = torch.nn.BCELoss(reduction='sum')
# score_fn = nn.L1Loss() if args.dataset in ["zinc", "zinc12", "zinc250"] else nn.MSELoss()

record_path = args.model + ('_full_graph' if args.full_graph else '') + ('_no_cluster' if args.no_cluster else '') + args.name + '_' + str(args.dataset) \
              + '/' + 'resolution' + str(args.resolution) + '_k_' + str(args.k) + '_distlimitu_' + str(args.distlimitu) + '_l_' + str(args.distlimitl) + '_pool_node_' + str(args.node_pool) + '_subgraph_' + str(args.subgraph_pool) + '_global_' + str(args.global_pool) + '_rate_' + str(args.rate) + '_perm_' + str(args.drop_perm) + '_mask_' + str(args.mask_value) + '_hid_dim_' + str(args.hid_dim) + '_layers_' + str(args.num_layers) + '_id_' + str(args.num_layers_id) + '_global_' + str(args.num_layers_global) + "_regression_" + str(args.num_layers_regression) + '_cat_' + args.cat + '_norm_' + args.norm_type \
              + '/lr_' + str(args.lr) + ('_cos-lr_' if args.cos_lr else '') + (('_Plateau_patience_' + str(args.patience) + '_factor_' + str(args.factor)) if not args.cos_lr else '') + '_decay_' + str(args.weight_decay) + '_epochs_' + str(args.epochs) + '_bs_' + str(args.batch_size) + ('_ensemble_train' if args.ensemble_train else '') + (('_ensemble_test_' + str(args.sample_times) + '_' + args.ensemble_mode) if args.ensemble_test else '') + '_dropout_' + str(args.drop_ratio) + '_no_' + str(args.no)
if not os.path.isdir(record_path):
    mkdir_p(record_path)
save_model_name = record_path + '/model.pkl'
save_curve_name = record_path + '/curve.pkl'

accuracy_file = record_path + '/test_MSE_epoch.txt'
record_file = record_path + '/training_process.txt'

print("Save model name:", save_model_name)
print("Save curve name:", save_curve_name)

training_configurations = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'initial_learning_rate': args.lr,
            'changing_lr': [100],
            'lr_decay_rate': 0.5,
            'momentum': 0.9,
            'nesterov': True,
            'weight_decay': args.weight_decay} # 1e-5


def buildMod(k, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, node_pool, subgraph_pool, global_pool, rate, ensemble_test, drop_perm, mask_value, cat, drop_ratio, norm, dataset):
    max_nodez, max_edgez = None, None
    in_dim = None
    if dataset.data.x is not None:
        max_nodez = torch.max(dataset.data.x) # types of atoms
        in_dim = dataset.data.x.shape[-1]
        print(in_dim)
        print("max_nodez", max_nodez)
        print("use node attr")
    max_edgez1 = torch.max(dataset.data.edge_attr[:, 0]) # types of edge
    max_edgez2 = torch.max(dataset.data.edge_attr[:, 1])
    max_edgez3 = torch.max(dataset.data.edge_attr[:, 2])
    max_edgez = max_edgez1
    print("max_edgez", max_edgez1, max_edgez2, max_edgez3)
    print("use edge attr")
    if args.model == 'Transformer':
        return IDMPNN_Transformer(k, in_dim, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, max_nodez, max_edgez1, max_edgez2, max_edgez3, node_pool, subgraph_pool, global_pool, rate, cat, drop_ratio, drop_perm, norm, ensemble_test, num_head=args.num_head)
    if args.model == 'IDMPNN':
        return IDMPNN_Global_parallel(k, in_dim, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, max_nodez, max_edgez1, max_edgez2, max_edgez3, node_pool, subgraph_pool, global_pool, rate, cat, drop_ratio, drop_perm, norm, ensemble_test)
    elif args.model == 'GINE':
        return ID_GINE(k, in_dim, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, max_nodez, max_edgez1, max_edgez2, max_edgez3, node_pool, subgraph_pool, global_pool, rate, cat, drop_ratio, drop_perm, norm, ensemble_test)
    elif args.model == 'IDMPNN_Discrete':
        return IDMPNN_Discrete(k, in_dim, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, max_nodez, max_edgez, node_pool, subgraph_pool, global_pool, rate, cat, drop_ratio, drop_perm, norm, ensemble_test)

def train(mod, opt: AdamW, dl_list):
    mod.train()
    losss = []
    N = 0
    train_idx = [4, 1, 3, 2, 0]
    for idx in train_idx:
        # print(len(dl))
        dl = dl_list[idx]
        for dat in dl:
            opt.zero_grad()
            x, subgs, edge_index, edge_attr, y, num_subg, num_node, num_edge = dat.x, dat.subgs, dat.edge_index, dat.edge_attr, dat.y, dat.num_subg, dat.num_node, dat.num_edge
            pred = mod(x, edge_index, edge_attr, subgs, num_subg, num_node, num_edge)
            mask = ~torch.isnan(y)
            y = dat.y.to(torch.float)[mask]
            pred = pred[mask]
            loss = loss_fn(pred.flatten(), y.flatten())
            loss.backward()
            opt.step()
            num_graphs = dat.num_graphs
            losss.append(loss)
            N += num_graphs
    losss = [_.item() for _ in losss]
    return np.sum(losss) / N


@torch.no_grad()
def test(mod, dl_list, evaluator):
    mod.eval()
    y_preds, y_trues = [], []
    for dl in dl_list:
        for dat in dl:
            x, subgs, edge_index, edge_attr, y, num_subg, num_node, num_edge = dat.x, dat.subgs, dat.edge_index, dat.edge_attr, dat.y, dat.num_subg, dat.num_node, dat.num_edge
            pred = mod(x, edge_index, edge_attr, subgs, num_subg, num_node, num_edge)
            y_preds.append(pred)
            y_trues.append(y)
    return evaluator.eval({
        'y_pred': torch.cat(y_preds, dim=0),
        'y_true': torch.cat(y_trues, dim=0),
    })[evaluator.eval_metric]


@torch.no_grad()
def test_ensemble(mod, dl_list, evaluator):
    mod.eval()
    y_preds, y_trues = [], []
    for dl in dl_list:
        for dat in dl:
            batch_pred = []
            for i in range(args.sample_times):
                x, subgs, edge_index, edge_attr, y, num_subg, num_node, num_edge = dat.x, dat.subgs, dat.edge_index, dat.edge_attr, dat.y, dat.num_subg, dat.num_node, dat.num_edge
                pred = mod(x, edge_index, edge_attr, subgs, num_subg, num_node, num_edge)
                batch_pred.append(pred.flatten())
            y = dat.y
            batch_pred = torch.cat(batch_pred).reshape(args.sample_times, -1)
            # print(batch_pred)
            if args.ensemble_mode == 'mean':
                pred = torch.mean(batch_pred, dim=0)
                y_preds.append(pred)
            elif args.ensemble_mode == 'best':
                best_pred = []
                for i in range(batch_pred.shape[1]):
                    # print(torch.argmin(abs(batch_pred[:, i] - y[i])))
                    best_pred.append(batch_pred[torch.argmin(abs(batch_pred[:, i] - y[i])), i])
                pred = torch.tensor(best_pred).reshape(-1, 1)
                y_preds.append(pred)
            else:
                pred = torch.median(batch_pred, dim=0)
                y_preds.append(pred[0].reshape(-1, 1))
            # print(pred)
            # print(y)
            y_trues.append(y)
    return evaluator.eval({
        'y_pred': torch.cat(y_preds, dim=0),
        'y_true': torch.cat(y_trues, dim=0),
    })[evaluator.eval_metric]


model = buildMod(k, args.hid_dim, args.out_dim, args.num_layers, args.num_layers_global, args.num_layers_id, args.num_layers_regression, args.node_pool, args.subgraph_pool, args.global_pool, args.rate, args.ensemble_test, args.drop_perm, args.mask_value, args.cat, args.drop_ratio, args.norm_type, dataset).to(device)
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # lr=3e-4
# scd = ReduceLROnPlateau(optimizer, mode="min", factor=args.factor, patience=args.patience)
scd = StepLR(optimizer, step_size=args.patience, gamma=args.factor)
train_loaders = [train_dataloader_1, train_dataloader_2, train_dataloader_3, train_dataloader_4, train_dataloader_5]
test_loaders = [test_dataloader_1, test_dataloader_2, test_dataloader_3, test_dataloader_4, test_dataloader_5]
val_loaders = [val_dataloader_1, val_dataloader_2, val_dataloader_3, val_dataloader_4, val_dataloader_5]
for i in range(args.epochs):
    if args.cos_lr:
        adjust_learning_rate(optimizer, i, args.cos_lr, training_configurations)
    t1 = time.time()
    loss = train(model, optimizer, train_loaders)
    t2 = time.time()
    # print("train end", flush=True)
    if args.ensemble_test:
        val_score = test_ensemble(model, val_loaders, evaluator)
    else:
        val_score = test(model, val_loaders, evaluator)
    scd.step()
    t3 = time.time()
    print(
        f"epoch {i}: train {loss:.4e} {int(t2 - t1)}s valid {val_score:.4e} {int(t3 - t2)}s ",
        end="", flush=True)
    if val_score < best_val_score:
        best_val_score = val_score
        torch.save(model.state_dict(), save_model_name)
    if args.ensemble_test:
        tst_score = test_ensemble(model, test_loaders, evaluator)
    else:
        tst_score = test(model, test_loaders, evaluator)
    t4 = time.time()
    print(f"tst {tst_score:.4e} {int(t4-t3)}s ", end="")
    print(optimizer.param_groups[0]['lr'])
    # print(flush=True)

    string = str({'Train': loss, 'Validation': val_score, 'Test': tst_score})
    fd = open(record_file, 'a+')
    fd.write(string + '\n')
    fd.close()

    train_curve.append(loss)
    valid_curve.append(val_score)
    test_curve.append(tst_score)

best_val_epoch = np.argmax(np.array(valid_curve))
best_train = min(train_curve)

print('Finished training!')
print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
print('Test score: {}'.format(test_curve[best_val_epoch]))
print('Best test: {}'.format(np.max(np.array(test_curve))))

np.savez(save_curve_name, train=np.array(train_curve), val=np.array(valid_curve), test=np.array(test_curve),
             test_for_best_val=test_curve[best_val_epoch])
np.savetxt(accuracy_file, np.array(test_curve))

string = 'Best validation score: ' + str(valid_curve[best_val_epoch]) + ' Test score: ' + str(test_curve[best_val_epoch])
mean_test_acc = np.mean(np.array(test_curve)[-10:-1])
fd = open(record_file, 'a+')
fd.write(string + '\n')
fd.write('mean test acc: ' + str(mean_test_acc) + '\n')
fd.close()

plt.figure()
plt.plot(test_curve, color='b')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Test MSE')
# plt.show()
plt.savefig(record_path + '/Test_MSE.png')