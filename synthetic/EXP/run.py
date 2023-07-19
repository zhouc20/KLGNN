import copy
import time
import numpy as np
import torch.nn as nn
import torch
# from IDMPNN import IDMPNN
from IDMPNN_Global import IDMPNN_Global, IDMPNN_Global_parallel, IDMPNN_Global_Local
from preprocess import graph2IDsubgraph_global, graph2IDsubgraph_global_new
from datalist import DataListSet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PlanarSATPairsDataset import PlanarSATPairsDataset
from torch_geometric.data import DataLoader, Data, Batch
import torch.nn.functional as F
import argparse
import random
import shutil
import tqdm
import os, sys
import math
import errno
import matplotlib.pyplot as plt
# from dataset import load_dataset


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
parser.add_argument("--task", type=int, default=0, help='assert in [0, 1, 2]')
parser.add_argument('--task_type', type=str, default='graph')
parser.add_argument("--dataset", type=str, default="EXP", help=" 'EXP' or 'CEXP' ")
parser.add_argument("--k", type=int, default=4)
parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--strategy', type=str, default='bfs')
parser.add_argument('--batch_size', type=int, default=1)
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
parser.add_argument('--num_layers_id', type=int, default=0)
parser.add_argument('--num_layers_global', type=int, default=1)
parser.add_argument('--num_layers_regression', type=int, default=1)
parser.add_argument('--pool', type=str, default='max')
parser.add_argument('--full_graph', action='store_true', default=False)
parser.add_argument('--rate', type=float, default=1.0)
parser.add_argument('--num_tasks', type=int, default=1)
parser.add_argument('--no', type=int, default=0)
parser.add_argument('--name', type=str, default='')
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--local', action='store_true', default=False)
parser.add_argument('--ensemble_train', action='store_true', default=False)
parser.add_argument('--ensemble_test', action='store_true', default=False)
parser.add_argument('--sample_times', type=int, default=1)
parser.add_argument('--ensemble_mode', type=str, default='mean')
parser.add_argument('--distlimit', type=int, default=-1)
parser.add_argument('--distlimitu', type=int, default=3)
parser.add_argument('--distlimitl', type=int, default=-1)
parser.add_argument('--factor', type=float, default=0.9)  # 0.5
parser.add_argument('--patience', type=int, default=5)  # 3
parser.add_argument('--split',type=int,default=10, help='number of fold in cross validation')

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


k = args.k
depth = args.depth
device = torch.device("cuda")


def pre_transform(g):
    return graph2IDsubgraph_global_new(g, args.k, 64, args.distlimitl, args.distlimitu)

def PyG_collate(examples):
    """PyG collcate function
    Args:
        examples(list): batch of samples
    """
    data = Batch.from_data_list(examples)
    return data

path = args.dataset
# if os.path.exists(path+'/processed') and args.reprocess:
#     shutil.rmtree(path+'/processed')

dataset = PlanarSATPairsDataset(path, pre_transform=pre_transform)
# val_dataset = PlanarSATPairsDataset(path, pre_transform=pre_transform)
# test_dataset = GraphPropertyDataset(path, split='test', pre_transform=pre_transform)

#additional parameter for EXP dataset and training
args.in_dim = 2
args.out_dim = dataset.num_classes
# print(dataset.num_classes)
args.MODULO = 4
args.MOD_THRESH = 1

#output argument to log file

tr_accuracies = np.zeros((args.epochs, args.split))
tst_accuracies = np.zeros((args.epochs, args.split))
tst_exp_accuracies = np.zeros((args.epochs, args.split))
tst_lrn_accuracies = np.zeros((args.epochs, args.split))
acc = []
tr_acc = []

record_path = 'IDMPNN_subgraph' + ('_full_graph' if args.full_graph else '') + ('_parallel' if args.parallel else '') + '/' + args.name + '_' + str(args.dataset) + '_task_' + str(args.task) + '_loss_' + str(args.loss) \
              + '_IDMPNN' + '_k_' + str(args.k) + '_distlimitu_' + str(args.distlimitu) + '_distlimitl_' + str(args.distlimitl) + '_pool_' + str(args.pool) + '_rate_' + str(args.rate) + '_hid_dim_' + str(args.hid_dim) + '_layers_' + str(args.num_layers) + '_layers_id_' + str(args.num_layers_id) + '_layers_global_' + str(args.num_layers_global) + "_layers_regression_" + str(args.num_layers_regression) \
              + '/lr_' + str(args.lr) + ('_cos-lr_' if args.cos_lr else '') + (('_Plateau_patience_' + str(args.patience) + '_factor_' + str(args.factor)) if not args.cos_lr else '') + '_epochs_' + str(args.epochs) + '_bs_' + str(args.batch_size) + ('_ensemble_train' if args.ensemble_train else '') + (('_ensemble_test_' + str(args.sample_times) + '_' + args.ensemble_mode) if args.ensemble_test else '') + '_no_' + str(args.no)
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

def buildMod(k, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, pool, parallel, local, rate, dataset):
    max_nodez, max_edgez = None, None
    in_dim = 1
    if parallel:
        return IDMPNN_Global_parallel(k, in_dim, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, num_layer_regression, max_nodez, max_edgez, pool, rate)
    elif local:
        return IDMPNN_Global_Local(k, in_dim, hid_dim, out_dim, num_layer, num_layer_global, num_layer_id, max_nodez, max_edgez, pool)
    else:
        return IDMPNN_Global(k, in_dim, hid_dim, out_dim, num_layer, num_layer_global, max_nodez, max_edgez, pool)


@torch.no_grad()
def val(data_loader, model, device):
    model.eval()
    loss_all = 0
    for batch_graphs in data_loader:
        dat = batch_graphs.to(device)
        batch_size=dat.num_graphs
        x, adj, subgs, subadj, y, pos, num_subg, num_node = dat.x, dat.adj, dat.subgs, dat.subadj, dat.y, dat.pos, dat.num_subg, dat.num_node
        predict=model(x, adj, subgs, num_subg, num_node)
        predict=F.log_softmax(predict,dim=-1)
        loss=F.nll_loss(predict,batch_graphs.y, reduction='sum').item()
        loss_all += loss

    model.train()
    return loss_all / len(data_loader.dataset)


@torch.no_grad()
def test(data_loader, model, device):
    model.eval()
    correct = 0
    for batch_graphs in data_loader:
        dat = batch_graphs.to(device)
        batch_size=dat.num_graphs
        x, adj, subgs, subadj, y, pos, num_subg, num_node = dat.x, dat.adj, dat.subgs, dat.subadj, dat.y, dat.pos, dat.num_subg, dat.num_node
        nb_trials = 1   # Support majority vote, but single trial is default
        successful_trials = torch.zeros_like(batch_graphs.y)
        for i in range(nb_trials):  # Majority Vote
            pred = model(x, adj, subgs, num_subg, num_node).max(1)[1]
            successful_trials += pred.eq(batch_graphs.y)
        successful_trials = successful_trials > (nb_trials // 2)
        correct += successful_trials.sum().item()

    model.train()
    return correct / len(data_loader.dataset)

print(len(dataset))
loader = DataLoader(dataset, batch_size=2, collate_fn=PyG_collate)
i = 0
correct = 0
for batch in loader:
    # print(batch)
    batch = batch.to(device)
    i += 1
    model = buildMod(k, args.hid_dim, args.out_dim, args.num_layers, args.num_layers_global, args.num_layers_id,
                     args.num_layers_regression, args.pool, args.parallel, args.local, args.rate, dataset).to(device)

    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=1e-8)
    for epoch in range(args.epochs):
        # print(f'Starting epoch {epoch + 1}...')
        # fd = open(record_file, 'a+')
        # fd.write(f'Starting epoch {epoch + 1}...' + '\n')
        #  fd.close()
        graphs = copy.deepcopy(batch)
        # print(graphs.x[0])
        # batch_size = graphs.num_graphs
        optimizer.zero_grad()
        x, adj, subgs, subadj, y, pos, num_subg, num_node = graphs.x, graphs.adj, graphs.subgs, graphs.subadj, graphs.y, graphs.pos, graphs.num_subg, graphs.num_node
        if not args.local:
            predict = model(x, adj, subgs, num_subg, num_node)
        else:
            predict = model(x, adj, subadj, subgs, num_subg, num_node)
        pred = predict.max(1)[1]
        success = pred.eq(y)
        predict = F.log_softmax(predict, dim=-1)
        loss = F.nll_loss(predict, graphs.y)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        model.zero_grad()

        loss_val = loss.item()
        lr = optimizer.param_groups[0]['lr']
        del graphs
        if (success.sum() == 2) or (epoch == args.epochs - 1):
            print(epoch, success)
            correct += success.sum()
            break
print(correct)
print(correct/len(dataset))

# for i in range(args.split):
#     print(f"---------------Training on fold {i}------------------------")
#     fd = open(record_file, 'a+')
#     fd.write(f"---------------Training on fold {i}------------------------" + '\n')
#     fd.close()
#     model = buildMod(k, args.hid_dim, args.out_dim, args.num_layers, args.num_layers_global, args.num_layers_id, args.num_layers_regression, args.pool, args.parallel, args.rate, dataset).to(device)
#
#     model.to(device)
#     model.train()
#     # pytorch_total_params = train_utils.count_parameters(model)
#     # log.info(f'The total parameters of model :{[pytorch_total_params]}')
#
#     optimizer = Adam(model.parameters(), lr=args.lr)
#     scheduler = ReduceLROnPlateau(
#         optimizer, mode='min', factor=args.factor, patience=args.patience, min_lr=1e-8)
#
#     # K-fold cross validation split
#     n = len(dataset) // args.split
#     test_mask = torch.zeros(len(dataset))
#     test_exp_mask = torch.zeros(len(dataset))
#     test_lrn_mask = torch.zeros(len(dataset))
#
#     test_mask[i * n:(i + 1) * n] = 1  # Now set the masks
#     learning_indices = [x for idx, x in enumerate(range(n * i, n * (i + 1))) if x % args.MODULO <= args.MOD_THRESH]
#     test_lrn_mask[learning_indices] = 1
#     exp_indices = [x for idx, x in enumerate(range(n * i, n * (i + 1))) if x % args.MODULO > args.MOD_THRESH]
#     test_exp_mask[exp_indices] = 1
#
#     # Now load the datasets
#     test_dataset = dataset[test_mask.bool()]
#     test_exp_dataset = dataset[test_exp_mask.bool()]
#     test_lrn_dataset = dataset[test_lrn_mask.bool()]
#     train_dataset = dataset[(1 - test_mask).bool()]
#
#     n = len(train_dataset) // args.split
#     val_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
#     val_mask[i * n:(i + 1) * n] = 1
#     val_dataset = train_dataset[val_mask]
#     train_dataset = train_dataset[~val_mask]
#
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=PyG_collate)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=PyG_collate)
#     test_exp_loader = DataLoader(test_exp_dataset, batch_size=args.batch_size,
#                                       collate_fn=PyG_collate)  # These are the new test splits
#     test_lrn_loader = DataLoader(test_lrn_dataset, batch_size=args.batch_size, collate_fn=PyG_collate)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=PyG_collate)
#     best_val_loss, best_test_acc, best_train_acc = 100, 0, 0
#     for epoch in range(args.epochs):
#         print(f'Starting epoch {epoch + 1}...')
#         fd = open(record_file, 'a+')
#         fd.write(f'Starting epoch {epoch + 1}...' + '\n')
#         fd.close()
#         for graphs in train_loader:
#             graphs = graphs.to(device)
#             # print(graphs.x[0])
#             batch_size = graphs.num_graphs
#             optimizer.zero_grad()
#             x, adj, subgs, subadj, y, pos, num_subg, num_node = graphs.x, graphs.adj, graphs.subgs, graphs.subadj, graphs.y, graphs.pos, graphs.num_subg, graphs.num_node
#             predict = model(x, adj, subgs, num_subg, num_node)
#             predict = F.log_softmax(predict, dim=-1)
#             loss = F.nll_loss(predict, graphs.y)
#             loss.backward()
#             # nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#             optimizer.step()
#             model.zero_grad()
#
#             loss_val = loss.item()
#             lr = optimizer.param_groups[0]['lr']
#
#         print(f"evaluate after epoch {epoch + 1}...")
#         train_loss = val(train_loader, model, device)
#         val_loss = val(val_loader, model, device)
#         scheduler.step(val_loss)
#         train_acc = test(train_loader, model, device)
#         test_acc = test(test_loader, model, device)
#         if best_val_loss >= val_loss:
#             best_val_loss = val_loss
#             best_train_acc = train_acc
#             best_test_acc = test_acc
#
#         test_exp_acc = test(test_exp_loader, model, device)
#         test_lrn_acc = test(test_lrn_loader, model, device)
#         tr_accuracies[epoch, i] = train_acc
#         tst_accuracies[epoch, i] = test_acc
#         tst_exp_accuracies[epoch, i] = test_exp_acc
#         tst_lrn_accuracies[epoch, i] = test_lrn_acc
#         string = 'Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, Val Loss: {:.7f}, Test Acc: {:.7f}, Exp Acc: {:.7f}, Lrn Acc: {:.7f}, Train Acc: {:.7f}'.format(
#             epoch + 1, lr, train_loss, val_loss, test_acc, test_exp_acc, test_lrn_acc, train_acc)
#         print(string)
#         fd = open(record_file, 'a+')
#         fd.write(string + '\n')
#         fd.close()
#
#     acc.append(best_test_acc)
#     tr_acc.append(best_train_acc)
# acc = torch.tensor(acc)
# tr_acc = torch.tensor(tr_acc)
# print("-------------------Print final result-------------------------")
# print(f"Train result: Mean: {tr_acc.mean().item()}, Std :{tr_acc.std().item()}")
# print(f"Test result: Mean: {acc.mean().item()}, Std :{acc.std().item()}")
# fd = open(record_file, 'a+')
# fd.write(f"Train result: Mean: {tr_acc.mean().item()}, Std :{tr_acc.std().item()}" + '\n')
# fd.write(f"Test result: Mean: {acc.mean().item()}, Std :{acc.std().item()}" + '\n')
# fd.close()

