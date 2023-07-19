import time
import numpy as np
import torch.nn as nn
import torch
# from IDMPNN import IDMPNN
from IDGPS import IDMPNN_Transformer
from IDMPNN_Global import IDMPNN_Global, IDMPNN_Global_parallel
from preprocess import graph2IDsubgraph_global, graph2IDsubgraph_cluster
from datalist import DataListSet
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader as PygDataloader
import argparse
import random
import os, sys
import math
import errno
import matplotlib.pyplot as plt
from dataset import load_dataset


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
parser.add_argument("--dataset", type=str, default="zinc12")
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
parser.add_argument('--attn_drop', type=float, default=0.0, help='drop out rate')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cos_lr', action='store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--loss', type=str, default='L1Loss')
parser.add_argument('--in_dim', type=int, default=16)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--out_dim', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_layers_id', type=int, default=0)
parser.add_argument('--num_layers_global', type=int, default=1)
parser.add_argument('--num_layers_regression', type=int, default=1)
parser.add_argument('--transformer', action='store_true', default=False)
parser.add_argument('--local_MPNN', action='store_true', default=False)
parser.add_argument('--central_encoding', action='store_true', default=False)
parser.add_argument('--attn_bias', action='store_true', default=False)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--rw_steps', type=int, default=20)
parser.add_argument('--se_dim', type=int, default=16)
parser.add_argument('--se_type', type=str, default='linear')
parser.add_argument('--norm_type', type=str, default='layer')
parser.add_argument('--cat', type=str, default='add')
parser.add_argument('--final_concat', type=str, default='none')
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
parser.add_argument('--discrete', action='store_true', default=False)
parser.add_argument('--ensemble_train', action='store_true', default=False)
parser.add_argument('--ensemble_test', action='store_true', default=False)
parser.add_argument('--sample_times', type=int, default=1)
parser.add_argument('--ensemble_mode', type=str, default='mean')
parser.add_argument('--distlimit', type=int, default=-1)
parser.add_argument('--distlimitu', type=int, default=3)
parser.add_argument('--distlimitl', type=int, default=-1)
parser.add_argument('--factor', type=float, default=0.9)  # 0.5
parser.add_argument('--patience', type=int, default=5)  # 3

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


k = args.k
depth = args.depth
strategy = args.strategy
assert strategy in ['neighbor', 'path', 'subgraph', 'bfs']
device = torch.device("cuda")

glist, trn_idx, val_idx, tst_idx = load_dataset(args.dataset, args.task)
print('load dataset!', flush=True)

print('Preprocessing...', flush=True)
t1 = time.time()
processed_path = f"data/{args.dataset}_{k}_{args.distlimitl}_{args.distlimitu}_res_{args.resolution}_rw_{args.rw_steps}.pt" if not args.no_cluster else f"data/{args.dataset}_{k}_{args.distlimitl}_{args.distlimitu}.pt"
if os.path.exists(processed_path):
    dataset = torch.load(processed_path, map_location="cpu")
elif not args.no_cluster:
    datalist = [
        graph2IDsubgraph_cluster(dat, k, max([g.x.shape[0] for g in glist]),
                                    args.distlimitl, args.distlimitu, args.resolution, args.rw_steps)
        for dat in glist
    ]
    dataset = DataListSet(datalist)
    torch.save(dataset, processed_path)
else:
    datalist = [
        graph2IDsubgraph_global(dat, k, max([g.x.shape[0] for g in glist]),
                                 args.distlimitl, args.distlimitu, args.rw_steps)
        for dat in glist
    ]
    dataset = DataListSet(datalist)
    torch.save(dataset, processed_path)
# processed_path = f"data/{args.dataset}_{k}_{args.distlimit}.pt"
# if os.path.exists(processed_path):
#     dataset = torch.load(processed_path, map_location="cpu")
# else:
#     datalist = [graph2IDsubgraph_global(dat, k, max([g.x.shape[0] for g in glist]), args.distlimit) for dat in glist]
#     dataset = DataListSet(datalist)
#     torch.save(dataset, processed_path)
dataset.data = dataset.data.to(device)
print(f"preprocess {int(time.time()-t1)} s", flush=True)
print(len(dataset))

loss_fn = nn.L1Loss() if args.dataset in ["zinc", "zinc12", "zinc250"] else nn.MSELoss()
score_fn = nn.L1Loss() if args.dataset in ["zinc", "zinc12", "zinc250"] else nn.MSELoss()

record_path = 'IDMPNN_parallel' + ('_full_graph' if args.full_graph else '') + ('_no_cluster' if args.no_cluster else '') + ('/bi_aggregate/' if args.bi_aggregate else '') + ('/aggregate/' if args.aggregate else '') + ('/AK+/' if args.ak else '') + ('/baseline/' if not args.bi_aggregate and not args.aggregate else '') + args.name + '_' + str(args.dataset) \
              + '_IDMPNN' + (('_GPS_' + str(args.num_head)) if args.transformer else '') + '_res' + str(args.resolution) + '_k_' + str(args.k) + '_distlimitu_' + str(args.distlimitu) + '_l_' + str(args.distlimitl) + ('_discrete_' if args.discrete else '') + '_pool_node_' + str(args.node_pool) + '_subgraph_' + str(args.subgraph_pool) + '_global_' + str(args.global_pool) + '_rate_' + str(args.rate) + '_perm_' + str(args.drop_perm) + '_mask_' + str(args.mask_value) + '_hid_dim_' + str(args.hid_dim) + '_layers_' + str(args.num_layers) + '_id_' + str(args.num_layers_id) + '_global_' + str(args.num_layers_global) + "_regression_" + str(args.num_layers_regression) + '_cat_' + args.cat + '_final_' + args.final_concat + '_norm_' + args.norm_type + '_' + str(args.rw_steps) + '_' + str(args.se_dim) + args.se_type \
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


def buildMod(k, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, node_pool, subgraph_pool, global_pool, bi_aggregate, aggregate, ak, rate, drop_perm, mask_value, cat, drop_ratio, norm, dataset):
    max_nodez, max_edgez = None, None
    in_dim = None
    if dataset.data.x is not None:
        max_nodez = torch.max(dataset.data.x) # types of atoms
        in_dim = dataset.data.x.shape[-1]
        print("max_nodez", max_nodez)
        print("use node attr")
    if dataset.data.adj.dtype == torch.long:
        max_edgez = torch.max(dataset.data.adj) # types of edge
        print("max_edgez", max_edgez)
        print("use edge attr")
    if args.transformer:
        return IDMPNN_Transformer(k, in_dim, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id,
                                  num_layer_regression, max_nodez, max_edgez, node_pool, subgraph_pool, global_pool,
                                  rate, cat, drop_ratio, args.attn_drop, drop_perm, norm, args.ensemble_test,
                                  args.final_concat, args.num_head, args.local_MPNN, args.central_encoding,
                                  args.attn_bias, args.rw_steps, args.se_dim, args.se_type)
    else:
        return IDMPNN_Global_parallel(k, in_dim, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, max_nodez, max_edgez, node_pool, subgraph_pool, global_pool, rate, cat, drop_ratio, drop_perm, norm, args.ensemble_test, args.final_concat, args.rw_steps, args.se_dim, args.se_type)


def train(mod, opt: AdamW, dl):
    mod.train()
    losss = []
    N = 0
    for dat in dl:
        opt.zero_grad()
        x, subgs, adj, y, num_subg, num_node, rwse = dat.x, dat.subgs, dat.adj, dat.y, dat.num_subg, dat.num_node, dat.rwse
        pred = mod(x, adj, subgs, num_subg, num_node, rwse)
        loss = loss_fn(pred.flatten(), y.flatten())
        loss.backward()
        opt.step()
        num_graphs = dat.num_graphs
        losss.append(loss * num_graphs)
        N += num_graphs
    losss = [_.item() for _ in losss]
    return np.sum(losss) / N


@torch.no_grad()
def test(mod, dl):
    mod.eval()
    losss = []
    N = 0
    for dat in dl:
        x, subgs, adj, y, num_subg, num_node, rwse = dat.x, dat.subgs, dat.adj, dat.y, dat.num_subg, dat.num_node, dat.rwse
        pred = mod(x, adj, subgs, num_subg, num_node, rwse)
        losss.append(score_fn(pred.flatten(), y.flatten()) * dat.num_graphs)
        N += dat.num_graphs
    losss = [_.item() for _ in losss]
    return np.sum(losss) / N


@torch.no_grad()
def test_ensemble(mod, dl):
    mod.eval()
    losss = []
    N = 0
    for dat in dl:
        batch_pred = []
        for i in range(args.sample_times):
            x, subgs, adj, y, num_subg, num_node, rwse = dat.x, dat.subgs, dat.adj, dat.y, dat.num_subg, dat.num_node, dat.rwse
            pred = mod(x, adj, subgs, num_subg, num_node, rwse)
            batch_pred.append(pred.flatten())
        y = dat.y
        batch_pred = torch.cat(batch_pred).reshape(args.sample_times, -1)
        if args.ensemble_mode == 'mean':
            pred = torch.mean(batch_pred, dim=0)
        else:
            pred = torch.median(batch_pred, dim=0)
        losss.append(score_fn(pred[0].flatten(), y.flatten()) * dat.num_graphs)
        N += dat.num_graphs
    losss = [_.item() for _ in losss]
    return np.sum(losss) / N


train_curve = []
valid_curve = []
test_curve = []
best_val_score = float("inf")
bs = args.batch_size
trn_dataloader = PygDataloader(dataset[trn_idx], batch_size=bs, shuffle=True, drop_last=True)
val_dataloader = PygDataloader(dataset[val_idx], batch_size=bs)
tst_dataloader = PygDataloader(dataset[tst_idx], batch_size=bs)

model = buildMod(k, args.hid_dim, args.out_dim, args.num_layers, args.num_layers_global, args.num_layers_id, args.num_layers_regression, args.node_pool, args.subgraph_pool, args.global_pool, args.bi_aggregate, args.aggregate, args.ak, args.rate, args.drop_perm, args.mask_value, args.cat, args.drop_ratio, args.norm_type, dataset).to(device)
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # lr=3e-4
scd = ReduceLROnPlateau(optimizer, mode="min", factor=args.factor, patience=args.patience)
for i in range(args.epochs):
    if args.cos_lr:
        adjust_learning_rate(optimizer, i, args.cos_lr, training_configurations)
    t1 = time.time()
    loss = train(model, optimizer, trn_dataloader)
    t2 = time.time()
    # print("train end", flush=True)
    if args.ensemble_test:
        val_score = test_ensemble(model, val_dataloader)
    else:
        val_score = test(model, val_dataloader)
    scd.step(val_score)
    t3 = time.time()
    print(
        f"epoch {i}: train {loss:.4e} {int(t2 - t1)}s valid {val_score:.4e} {int(t3 - t2)}s ",
        end="", flush=True)
    if val_score < best_val_score:
        best_val_score = val_score
        torch.save(model.state_dict(), save_model_name)
    if args.ensemble_test:
        tst_score = test_ensemble(model, tst_dataloader)
    else:
        tst_score = test(model, tst_dataloader)
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

best_val_epoch = np.argmin(np.array(valid_curve))
best_train = min(train_curve)

print('Finished training!')
print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
print('Test score: {}'.format(test_curve[best_val_epoch]))

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