import errno
import time
import numpy as np
import torch.nn as nn
from dgl.data.utils import load_graphs
import torch
from torch_geometric.data import Data as PygData
from IDMPNN import graph2IDsubgraph, IDMPNN
from datalist import DataListSet
from torch.optim import Adam
from torch_geometric.data import DataLoader as PygDataloader
import argparse
import random
import os, sys
import math
import matplotlib.pyplot as plt

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
parser.add_argument("--dataset", type=str, default="dataset1")

parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
# parser.add_argument('--gnn', type=str, default='lrp_pure',
#                     help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
# parser.add_argument('--num_layer', type=int, default=5,
#                     help='number of GNN message passing layers (default: 5)')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cos_lr', action='store_true', default=False)
parser.add_argument('--bn', action='store_true', default=False)
parser.add_argument('--mlp', action='store_true', default=False)
parser.add_argument('--full_permutation', action='store_true', default=False)
parser.add_argument('--in_dim', type=int, default=16)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--out_dim', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=1)
# parser.add_argument('--edge_hidden_feats', type=int, default=128)
parser.add_argument('--base_encoder_type', type=str, default='MPNN')
parser.add_argument('--num_step_message_passing', type=int, default=3)
parser.add_argument('--hop', type=int, default=1)
parser.add_argument('--neighbor_truncated', type=int, default=20)
parser.add_argument('--k_truncated', type=int, default=4)
parser.add_argument('--num_tasks', type=int, default=1)
# parser.add_argument('--root_specific', action='store_true', default=False)
parser.add_argument('--no', type=int, default=1)

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


k = args.k_truncated
device = torch.device("cuda")

glist, y = load_graphs(f"data/{args.dataset}.bin")
y = y[args.task].to(device)
with open(f"data/{args.dataset}_train.txt") as f:
    trn_idx = [int(line) for line in f.readlines()]
with open(f"data/{args.dataset}_val.txt") as f:
    val_idx = [int(line) for line in f.readlines()]
with open(f"data/{args.dataset}_test.txt") as f:
    tst_idx = [int(line) for line in f.readlines()]

t1 = time.time()
datalist = [
    graph2IDsubgraph(
        PygData(edge_index=torch.stack(glist[i].edges(), dim=0),
                num_nodes=glist[i].number_of_nodes()), #.to(dev)
                k)
    for i in range(len(glist))
]
dataset = DataListSet(datalist)
dataset.data = dataset.data.to(device)
print(f"preprocess {int(time.time()-t1)}s", flush=True)

training_configurations = {
            'epochs': args.epochs,
            'batch_size': 1,
            'initial_learning_rate': args.lr, # default=0.1
            'changing_lr': [20, 40],
            'lr_decay_rate': 0.1,
            'momentum': 0.9,
            'nesterov': True,
            'weight_decay': 1e-5}

loss_fn = nn.MSELoss()
score_fn = nn.L1Loss()
# IDMPNN(4, 16, 16, 1, 2)
model = IDMPNN(k, args.in_dim, args.hid_dim, args.out_dim, args.num_layers).to(device)
optimizer = Adam(model.parameters(), lr=args.lr)
model = model.to(device)

record_path = 'IDMPNN/test_' + 'random_seed_' + args.task + '_' + str(args.dataset) + '_lr_' + str(args.lr) + ('_cos-lr_' if args.cos_lr else '') + '_epochs_' + str(args.epochs) \
              + '_IDMPNN' + '_k_' + str(args.k_truncated) + '_in_dim_' + str(args.in_dim) + '_hid_dim_' + str(args.hid_dim) + '_layers_' + str(args.num_layers) + '_no_' + str(args.no)
if not os.path.isdir(record_path):
    mkdir_p(record_path)
save_model_name = record_path + '/model.pkl'
save_curve_name = record_path + '/curve.pkl'

accuracy_file = record_path + '/test_MAE_epoch.txt'
record_file = record_path + '/training_process.txt'

print("Save model name:", save_model_name)
print("Save curve name:", save_curve_name)


def train(mod, opt: Adam, dataset, idx):
    mod.train()
    losses = []
    for i in idx:
        opt.zero_grad()
        subgs, subadj = dataset[i].subgs, dataset[i].subadj
        pred = mod(None, subadj, subgs)
        loss = loss_fn(pred.flatten(), y[i].flatten())
        loss.backward()
        opt.step()
        losses.append(loss)
    losses = [_.item() for _ in losses]
    return np.average(losses)


@torch.no_grad()
def test(mod, dataset, idx):
    mod.eval()
    losss = []
    for i in idx:
        subgs, subadj = dataset[i].subgs, dataset[i].subadj
        pred = mod(None, subadj, subgs)
        losss.append(score_fn(pred.flatten(), y[i].flatten()))
    losss = [_.item() for _ in losss]
    return np.average(losss)


train_curve = []
valid_curve = []
test_curve = []

best_val_score = float("inf")
for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch, args.cos_lr, training_configurations)
    t1 = time.time()
    random.shuffle(trn_idx)
    loss = train(model, optimizer, dataset, trn_idx)
    t2 = time.time()
    val_score = test(model, dataset, val_idx)
    t3 = time.time()
    print(
        f"epoch {epoch}: train {loss:.4e} {int(t2-t1)}s valid {val_score:.4e} {int(t3-t2)}s ",
        end="")
    if val_score < best_val_score:
        best_val_score = val_score
        torch.save(model.state_dict(), save_model_name)
    tst_score = test(model, dataset, tst_idx)
    for i in tst_idx:
        subgs, subadj = dataset[i].subgs, dataset[i].subadj
        pred = model(None, subadj, subgs)
        print(pred, y[i])
        break
    t4 = time.time()
    print(f"test {tst_score:.4e} {int(t4-t3)}s ", end="")
    print()
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



