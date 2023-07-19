import time
import numpy as np
import torch.nn as nn
import torch
# from IDMPNN import IDMPNN
from IDMPNN import IDMPNN_full
from IDPPGN import IDPPGN
from preprocess import graph2IDsubgraph_global_new
from datalist import DataListSet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader as PygDataloader
from torch_geometric.data import Data, Batch
import argparse
import random
import os, sys
import math
import errno
import matplotlib.pyplot as plt
from SRDataset import SRDataset


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
parser.add_argument("--dataset", type=str, default="zinc")
parser.add_argument('--model', type=str, default='MPNN')
parser.add_argument("--k", type=int, default=4)
parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--strategy', type=str, default='bfs')
parser.add_argument('--no_perm', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cos_lr', action='store_true', default=False)
parser.add_argument('--loss', type=str, default='L1Loss')
parser.add_argument('--in_dim', type=int, default=16)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--out_dim', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_layers_id', type=int, default=4)
parser.add_argument('--num_layers_global', type=int, default=1)
parser.add_argument('--num_layers_regression', type=int, default=4)
parser.add_argument('--cat', type=str, default='hadamard_product')
parser.add_argument('--pool', type=str, default='max')
parser.add_argument('--node_pool', type=str, default='add')
parser.add_argument('--full_graph', action='store_true', default=False)
parser.add_argument('--node_base', action='store_true', default=False)
parser.add_argument('--rate', type=float, default=1.0)
parser.add_argument('--num_tasks', type=int, default=1)
parser.add_argument('--no', type=int, default=0)
parser.add_argument('--name', type=str, default='')
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--ensemble_train', action='store_true', default=False)
parser.add_argument('--ensemble_test', action='store_true', default=False)
parser.add_argument('--sample_times', type=int, default=1)
parser.add_argument('--ensemble_mode', type=str, default='mean')
parser.add_argument('--distlimit', type=int, default=-1)
parser.add_argument('--distlimitu', type=int, default=3)
parser.add_argument('--distlimitl', type=int, default=-1)
parser.add_argument('--factor', type=float, default=0.9)  # 0.5
parser.add_argument('--patience', type=int, default=20)  # 3

args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, cos_lr, training_configurations):
    """Sets the learning rate"""

    if not cos_lr:
        if epoch in training_configurations['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations['lr_decay_rate']

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


def PyG_collate(examples):
    """PyG collcate function
    Args:
        examples(list): batch of samples
    """
    data=Batch.from_data_list(examples)
    return data

k = args.k
depth = args.depth
strategy = args.strategy
assert strategy in ['neighbor', 'path', 'subgraph', 'bfs']
device = torch.device("cuda")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

path="data/" + args.dataset
dataset = SRDataset(path, k=k)
# dataset.data.x = dataset.data.x.long()
dataset.data.y = torch.arange(len(dataset.data.y)).long() # each graph is a unique class
dataset = dataset
train_dataset = dataset
val_dataset = dataset
test_dataset = dataset

# glist, trn_idx, val_idx, tst_idx = load_dataset(args.dataset, args.task)
# print('load dataset!', flush=True)

t1 = time.time()
# processed_path = f"data/{args.dataset}_{k}_{args.distlimitl}_{args.distlimitu}.pt"
# if os.path.exists(processed_path):
#     dataset = torch.load(processed_path, map_location="cpu")
# else:
#     datalist = [
#         graph2IDsubgraph_global_new(dat, k, max([g.x.shape[0] for g in glist]),
#                                 args.distlimitl, args.distlimitu)
#         for dat in glist
#     ]
#     dataset = DataListSet(datalist)
#     torch.save(dataset, processed_path)
dataset.data = dataset.data.to(device)
print(f"preprocess {int(time.time()-t1)} s", flush=True)

# loss_fn = nn.L1Loss() if args.dataset in ["zinc"] else nn.MSELoss()
# score_fn = nn.L1Loss() if args.dataset in ["zinc"] else nn.MSELoss()

record_path = args.model + args.name + '_' + str(args.dataset) \
              + '/k_' + str(args.k) + '_depth_' + str(args.depth) + '_pool_' + str(args.pool) + '_node_pool_' + str(args.node_pool) + '_rate_' + str(args.rate) + '_cat_' + args.cat + '_hid_dim_' + str(args.hid_dim) + '_layers_' + str(args.num_layers) + '_idlayers_' + str(args.num_layers_id) + '_globallayers_' + str(args.num_layers_global) + '_full_' + str(args.full_graph) + '_parallel_' + str(args.parallel) \
              + '/lr_' + str(args.lr) + ('_cos-lr_' if args.cos_lr else '') + (('_Plateau_patience_' + str(args.patience) + '_factor_' + str(args.factor)) if not args.cos_lr else '') + '_epochs_' + str(args.epochs) + '_bs_' + str(args.batch_size) + ('_ensemble_train' if args.ensemble_train else '') + (('_ensemble_test_' + str(args.sample_times) + '_' + args.ensemble_mode) if args.ensemble_test else '') + '_no_' + str(args.no)

# record_path = 'IDMPNN_subgraph' + ('_full_graph' if args.full_graph else '') + ('/parallel/' if args.parallel else '') + args.name + '_' + str(args.dataset) \
#               + '_IDMPNN' + '_k_' + str(args.k) + '_distlimitu_' + str(args.distlimitu) + '_distlimitl_' + str(args.distlimitl) + '_pool_' + str(args.pool) + '_rate_' + str(args.rate) + '_hid_dim_' + str(args.hid_dim) + '_layers_' + str(args.num_layers) + '_layers_id_' + str(args.num_layers_id) + '_layers_global_' + str(args.num_layers_global) \
#               + '/lr_' + str(args.lr) + ('_cos-lr_' if args.cos_lr else '') + (('_Plateau_patience_' + str(args.patience) + '_factor_' + str(args.factor)) if not args.cos_lr else '') + '_epochs_' + str(args.epochs) + '_bs_' + str(args.batch_size) + ('_ensemble_train' if args.ensemble_train else '') + (('_ensemble_test_' + str(args.sample_times) + '_' + args.ensemble_mode) if args.ensemble_test else '') + '_no_' + str(args.no)
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
            'weight_decay': 1e-5}

def buildMod(model, k, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, cat, pool, node_pool, full_graph, node_base, rate, dataset):
    max_nodez, max_edgez = None, None
    in_dim = 1
    if model == 'MPNN':
        return IDMPNN_full(k, in_dim, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, max_nodez, max_edgez, node_pool, pool, rate)
    elif model == 'PPGN':
        return IDPPGN(k, in_dim, hid_dim, out_dim, num_layer, rate=rate, max_edgez=max_edgez, cat=cat)
    else:
        raise NotImplementedError

def train(mod, opt: Adam, dl):
    mod.train()
    # losss = []
    total_loss = 0
    for dat in dl:
        opt.zero_grad()
        x, subgs, adj, subadj, y, num_subg, num_node = dat.x, dat.subgs, dat.adj, dat.subadj, dat.y, dat.num_subg, dat.num_node
        if args.model == 'MPNN':
            pred = mod(x, adj, subgs, num_subg)
        else:
            pred = mod(x, adj, subgs, num_subg, num_node)
        loss = torch.nn.CrossEntropyLoss()(pred, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * dat.num_graphs
    return total_loss / len(dl.dataset)


@torch.no_grad()
def test(mod, dl):
    mod.eval()
    y_preds, y_trues = [], []
    for dat in dl:
        x, subgs, adj, subadj, y, num_subg, num_node = dat.x, dat.subgs, dat.adj, dat.subadj, dat.y, dat.num_subg, dat.num_node
        if args.model == 'MPNN':
            pred = mod(x, adj, subgs, num_subg)
        else:
            pred = mod(x, adj, subgs, num_subg, num_node)
        y_preds.append(torch.argmax(pred, dim=-1))
        y_trues.append(y)
    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    return (y_preds == y_trues).float().mean()


train_curve = []
valid_curve = []
test_curve = []
best_val_score = float("inf")
bs = args.batch_size

model = buildMod(args.model, k, args.hid_dim, len(dataset), args.num_layers, args.num_layers_global, args.num_layers_id, args.num_layers_regression, args.cat, args.pool, args.node_pool, args.full_graph, args.node_base, args.rate, dataset).to(device)
optimizer = Adam(model.parameters(), lr=args.lr)  # lr=3e-4
scd = ReduceLROnPlateau(optimizer, mode="max", factor=args.factor, patience=args.patience, min_lr=1e-9)

train_loader = PygDataloader(train_dataset,args.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=PyG_collate)
val_loader = PygDataloader(val_dataset,args.batch_size,shuffle=False, num_workers=args.num_workers,collate_fn=PyG_collate)
test_loader = PygDataloader(test_dataset,args.batch_size,shuffle=False, num_workers=args.num_workers,collate_fn=PyG_collate)

for i in range(args.epochs):
    if args.cos_lr:
        adjust_learning_rate(optimizer, i, args.cos_lr, training_configurations)
    t1 = time.time()
    loss = train(model, optimizer, train_loader)
    t2 = time.time()
    val_score = test(model, val_loader)
    scd.step(val_score)
    t3 = time.time()
    print(
        f"epoch {i}: train {loss:.4e} {int(t2 - t1)}s valid {val_score:.4e} {int(t3 - t2)}s ",
        end="", flush=True)
    if val_score < best_val_score:
        best_val_score = val_score
        torch.save(model.state_dict(), save_model_name)
    tst_score = val_score
    if i == 2000 and tst_score < 0.1:
        break
    t4 = time.time()
    print(f"tst {tst_score:.4e} {int(t4-t3)}s ", end="")
    # print(optimizer.param_groups[0]['lr'])
    print(flush=True)

    string = str({'Train': loss, 'Validation': val_score, 'Test': tst_score})
    fd = open(record_file, 'a+')
    fd.write(string + '\n')
    fd.close()

    train_curve.append(loss)
    valid_curve.append(val_score.cpu())
    test_curve.append(tst_score.cpu())
    if i % 1000 == 0:
        test(model, test_loader)

tst_score = test(model, test_loader)


best_val_epoch = np.argmax(np.array(valid_curve))
best_train = max(train_curve)

print('Finished training!')
print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
print('Test score: {}'.format(test_curve[best_val_epoch]))

