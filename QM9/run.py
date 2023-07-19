import time
import numpy as np
import torch.nn as nn
import torch
# from IDMPNN import IDMPNN
from IDMPNN_Global import IDMPNN_Global_new, IDMPNN_Global_parallel, IDMPNN_parallel_aggregate, IDMPNN_parallel_biaggregate, IDMPNN_parallel_AK
from IDPPGN import IDPPGN
from preprocess import graph2IDsubgraph, graph2IDsubgraph_global, graph2IDsubgraph_global_new, graph2IDsubgraph_cluster
from datalist import DataListSet
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch_geometric.data import DataLoader as PygDataloader
from torch_geometric.data import DataLoader, Data, Batch
import torch_geometric.transforms as T
import argparse
import random
import os, sys
import math
import errno
import matplotlib.pyplot as plt
from QM9Dataset import QM9, conversion


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
parser.add_argument("--dataset", type=str, default="QM9")
parser.add_argument('--task', type=int, default=0, choices=list(range(19)), help='number of task')
parser.add_argument('--model', type=str, default='MPNN')
parser.add_argument("--no_cluster", action='store_true', default=False)
parser.add_argument("--k", type=int, default=4)
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
parser.add_argument('--min_lr', type=float, default=1e-7)
parser.add_argument('--cos_lr', action='store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--loss', type=str, default='L1Loss')
parser.add_argument('--use_pos', action='store_true', default=False)
parser.add_argument('--in_dim', type=int, default=14)  # 11(x) + 3(pos)
parser.add_argument('--hid_dim', type=int, default=32)
parser.add_argument('--out_dim', type=int, default=1)  # 19 tasks
parser.add_argument('--num_layers', type=int, default=0)
parser.add_argument('--num_layers_id', type=int, default=0)
parser.add_argument('--num_layers_global', type=int, default=0)
parser.add_argument('--num_layers_regression', type=int, default=0)
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
parser.add_argument('--factor', type=float, default=0.8)  # 0.5
parser.add_argument('--patience', type=int, default=7)  # 3
parser.add_argument('--convert', type=str, default='post',
                        help='if "post", convert units after optimization; if "pre", \
                        convert units before optimization')
parser.add_argument('--subset', action='store_true', default=False)

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


def pre_transform(g):
    return graph2IDsubgraph_global_new(g, args.k, 29, args.distlimitl, args.distlimitu)

def pre_transform_cluster(g):
    return graph2IDsubgraph_cluster(g, args.k, 29, args.distlimitl, args.distlimitu, args.resolution)

class TaskTransform(object):
    def __init__(self, target, pre_convert=False):
        self.target = target
        self.pre_convert = pre_convert

    def __call__(self, data):
        data.y = data.y[:, int(self.target)]  # Specify target: 0 = mu
        if self.pre_convert:  # convert back to original units
            data.y = data.y / conversion[int(self.target)]
        return data


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes > 6


def PyG_collate(examples):
    """PyG collcate function
    Args:
        examples(list): batch of samples
    """
    data = Batch.from_data_list(examples)
    return data

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

k = args.k
depth = args.depth
strategy = args.strategy
assert strategy in ['neighbor', 'path', 'subgraph', 'bfs']
device = torch.device("cuda")

# glist, trn_idx, val_idx, tst_idx = load_dataset(args.dataset, args.task)
path = args.dataset

t1 = time.time()
processed_path = (path + f"/{args.dataset}_{k}_{args.distlimitl}_{args.distlimitu}.pt") if args.no_cluster else (path + f"/{args.dataset}_{k}_{args.distlimitl}_{args.distlimitu}_res_{args.resolution}.pt")
if os.path.exists(processed_path):
    dataset = torch.load(processed_path, map_location="cpu")
elif args.no_cluster:
    dataset = QM9(path, pre_transform=pre_transform, pre_filter=MyFilter())
    torch.save(dataset, processed_path)
else:
    dataset = QM9(path, pre_transform=pre_transform_cluster, pre_filter=MyFilter())
    torch.save(dataset, processed_path)
# processed_path = f"data/{args.dataset}_{k}_{args.distlimit}.pt"
# if os.path.exists(processed_path):
#     dataset = torch.load(processed_path, map_location="cpu")
# else:
#     datalist = [graph2IDsubgraph_global(dat, k, max([g.x.shape[0] for g in glist]), args.distlimit) for dat in glist]
#     dataset = DataListSet(datalist)
#     torch.save(dataset, processed_path)
dataset.data = dataset.data.to(device)
print('load dataset!', flush=True)
print(f"preprocess {int(time.time()-t1)} s", flush=True)
print(len(dataset))

dataset = dataset.shuffle()
# print(dataset.data.y[:10])

tenpercent = int(len(dataset) * 0.1)
mean = dataset.data.y[tenpercent:].mean(dim=0)
std = dataset.data.y[tenpercent:].std(dim=0)
dataset.data.y = (dataset.data.y - mean) / std
# print(mean, std)
# print(dataset.data.y[:10])

test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:2 * tenpercent]
train_dataset = dataset[2 * tenpercent:]

if args.subset:
    test_dataset = test_dataset[:int(tenpercent/3)]
    val_dataset = dataset[:int(tenpercent/3)]
    train_dataset = dataset[:int(8 * tenpercent / 3)]

loss_fn = nn.L1Loss() if args.dataset in ["QM9"] else nn.MSELoss()
score_fn = nn.L1Loss() if args.dataset in ["QM9"] else nn.MSELoss()

record_path = args.model + args.name + '_' + str(args.dataset) + ('_subset' if args.subset else '') + '/task_' + str(args.task) + ('_use_pos' if args.use_pos else '')\
              + '/' + ('no_cluster' if args.no_cluster else ('resolution_' + str(args.resolution))) + '_k_' + str(args.k) + '_distlimitu_' + str(args.distlimitu) + '_distlimitl_' + str(args.distlimitl) + '_node_pool_' + str(args.node_pool) + '_subgraph_pool_' + str(args.subgraph_pool) + '_global_pool_' + str(args.global_pool) + '_rate_' + str(args.rate) + '_perm_' + str(args.drop_perm) + '_mask_' + str(args.mask_value) + '_hid_dim_' + str(args.hid_dim) + '_layers_' + str(args.num_layers) + '_layers_id_' + str(args.num_layers_id) + '_layers_global_' + str(args.num_layers_global) + "_layers_regression_" + str(args.num_layers_regression) + '_cat_' + args.cat \
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


def buildMod(base_encoder, k, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, node_pool, subgraph_pool, global_pool, rate, drop_perm, mask_value, cat, drop_ratio, dataset):
    max_nodez, max_edgez = None, None
    in_dim = 14 if args.use_pos else 11
    # if dataset.data.x is not None:
    #     max_nodez = torch.max(dataset.data.x) # types of atoms
    #     in_dim = dataset.data.x.shape[-1]
    #     print("max_nodez", max_nodez)
    #     print("use node attr")
    if dataset.data.adj.dtype == torch.long:
        max_edgez = torch.max(dataset.data.adj) # types of edge
        print("max_edgez", max_edgez)
        print("use edge attr")
    if base_encoder == 'MPNN':
        return IDMPNN_Global_parallel(k, in_dim, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, max_nodez, max_edgez, node_pool, subgraph_pool, global_pool, rate, cat, drop_ratio, drop_perm)
    elif base_encoder == 'PPGN':
        return IDPPGN(k, in_dim, hid_dim, out_dim, num_layer, rate=rate, cat=cat, max_edgez=max_edgez)
    else:
        raise NotImplementedError


def train(mod, opt: AdamW, dl):
    mod.train()
    losss = []
    for dat in dl:
        opt.zero_grad()
        x, subgs, adj, y, num_subg, num_node, pos = dat.x, dat.subgs, dat.adj, dat.y[:, args.task], dat.num_subg, dat.num_node, dat.pos
        if args.use_pos:
            x = torch.cat([x, pos], dim=-1)
        pred = mod(x, adj, subgs, num_subg, num_node)
        # print(pred)
        # print(y)
        loss = loss_fn(pred.flatten(), y.flatten())
        loss.backward()
        opt.step()
        losss.append(loss * dat.num_graphs)
    losss = [_.item() for _ in losss]
    mean_loss = np.sum(losss) / len(train_dataset)
    return mean_loss / conversion[int(args.task)] if args.convert == 'post' else mean_loss


@torch.no_grad()
def test(mod, dl, std):
    mod.eval()
    losss = []
    for dat in dl:
        x, subgs, adj, y, num_subg, num_node, pos = dat.x, dat.subgs, dat.adj, dat.y[:, args.task], dat.num_subg, dat.num_node, dat.pos
        if args.use_pos:
            x = torch.cat([x, pos], dim=-1)
        pred = mod(x, adj, subgs, num_subg, num_node)
        losss.append(score_fn(pred.flatten() * std.cuda(), y.flatten() * std.cuda()) * dat.num_graphs)
    losss = [_.item() for _ in losss]
    mean_loss = np.sum(losss) / len(test_dataset)
    return mean_loss / conversion[int(args.task)] if args.convert == 'post' else mean_loss


@torch.no_grad()
def test_ensemble(mod, dl, std):
    mod.eval()
    losss = []
    for dat in dl:
        batch_pred = []
        for i in range(args.sample_times):
            x, subgs, adj, y, num_subg, num_node, pos = dat.x, dat.subgs, dat.adj, dat.y[:, args.task], dat.num_subg, dat.num_node, dat.pos
            if args.use_pos:
                x = torch.cat([dat.x, dat.pos], dim=-1)
                # print(x.shape)
                # print(adj.shape)
            pred = mod(x, adj, subgs, num_subg, num_node)
            batch_pred.append(pred.flatten())
        y = dat.y[:, args.task]
        batch_pred = torch.cat(batch_pred).reshape(args.sample_times, -1)
        if args.ensemble_mode == 'mean':
            pred = torch.mean(batch_pred, dim=0)
        else:
            pred = torch.median(batch_pred, dim=0)
        losss.append(score_fn(pred[0].flatten() * std.cuda(), y.flatten() * std.cuda()) * dat.num_graphs)
    losss = [_.item() for _ in losss]
    mean_loss = np.sum(losss) / len(test_dataset)
    return mean_loss / conversion[int(args.task)] if args.convert == 'post' else mean_loss


train_curve = []
valid_curve = []
test_curve = []
best_val_score = float("inf")
bs = args.batch_size
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=PyG_collate)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=PyG_collate)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PyG_collate)

model = buildMod(args.model, k, args.hid_dim, args.out_dim, args.num_layers, args.num_layers_global, args.num_layers_id, args.num_layers_regression, args.node_pool, args.subgraph_pool, args.global_pool, args.rate, args.drop_perm, args.mask_value, args.cat, args.drop_ratio, dataset).to(device)
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # lr=3e-4
scd = ReduceLROnPlateau(optimizer, mode="min", factor=args.factor, patience=args.patience)
for i in range(args.epochs):
    if args.cos_lr:
        adjust_learning_rate(optimizer, i, args.cos_lr, training_configurations)
    t1 = time.time()
    loss = train(model, optimizer, train_loader) * std[args.task]
    t2 = time.time()
    # print("train end", flush=True)
    if args.ensemble_test:
        val_score = test_ensemble(model, val_loader, std[args.task])
    else:
        val_score = test(model, val_loader, std[args.task])
    scd.step(val_score)
    t3 = time.time()
    print(
        f"epoch {i}: train {loss:.4e} {int(t2 - t1)}s valid {val_score:.4e} {int(t3 - t2)}s ",
        end="", flush=True)
    if val_score < best_val_score:
        best_val_score = val_score
        torch.save(model.state_dict(), save_model_name)
    if args.ensemble_test:
        tst_score = test_ensemble(model, test_loader, std[args.task])
    else:
        tst_score = test(model, test_loader, std[args.task])
    t4 = time.time()
    print(f"tst {tst_score:.4e} {int(t4-t3)}s ", end="")
    # print(flush=True)

    string = str({'Train': loss, 'Validation': val_score, 'Test': tst_score})
    fd = open(record_file, 'a+')
    fd.write(string + '\n')
    fd.close()

    train_curve.append(loss)
    valid_curve.append(val_score)
    test_curve.append(tst_score)
    print(optimizer.param_groups[0]['lr'])
    if optimizer.param_groups[0]['lr'] < args.min_lr:
        break

best_val_epoch = np.argmin(np.array(valid_curve))
best_train = min(train_curve)

print('Finished training!')
print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
print('Test score: {}'.format(test_curve[best_val_epoch]))

# np.savez(save_curve_name, train=np.array(train_curve), val=np.array(valid_curve), test=np.array(test_curve),
#              test_for_best_val=test_curve[best_val_epoch])
# np.savetxt(accuracy_file, np.array(test_curve))

# string = 'Best validation score: ' + str(valid_curve[best_val_epoch]) + ' Test score: ' + str(test_curve[best_val_epoch])
# mean_test_acc = np.mean(np.array(test_curve)[-10:-1])
# fd = open(record_file, 'a+')
# fd.write(string + '\n')
# fd.write('mean test acc: ' + str(mean_test_acc) + '\n')
# fd.close()
#
# plt.figure()
# plt.plot(test_curve, color='b')
# plt.xlabel('Epoch')
# plt.ylabel('MSE')
# plt.title('Test MSE')
# # plt.show()
# plt.savefig(record_path + '/Test_MSE.png')