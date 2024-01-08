# GNN/GIN Notebook
# Matthew McEneaney
# 6/22/21

from __future__ import absolute_import, division, print_function

# ROOT imports
import uproot as ur
import uproot3 as ur3

# ML Imports
import awkward as ak
import numpy as np
from numpy import ma
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib as mpl
import matplotlib.pyplot as plt

# Graph Learning DGL Imports
import dgl
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import makedirs, save_info, load_info
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
import torch.nn.functional as F

# PyTorch Ignite Imports
from ignite.engine import Engine, Events, EventEnum, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import global_step_from_engine
from ignite.contrib.handlers.base_logger import (
    BaseLogger,
    BaseOptimizerParamsHandler,
    BaseOutputHandler,
    BaseWeightsHistHandler,
    BaseWeightsScalarHandler,
)

# Utility imports
import math, os, datetime

# Load notebook extensions
# %load_ext tensorboard

# Define dataset class

class LambdasDataset(DGLDataset):
    _url = None
    _sha1_str = None
    mode = "mode"
    num_classes = 2
    dataset = None

    def __init__(self, name, dataset=None, raw_dir=None, force_reload=False, verbose=False):
        self.dataset = dataset
        super(LambdasDataset, self).__init__(name=name,
                                          url=self._url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def process(self):
        mat_path = os.path.join(self.raw_path,self.mode+'_dgl_graph.bin')
        # process data to a list of graphs and a list of labels
        if self.dataset != None:
            self.graphs, self.labels = self.dataset["data"], torch.LongTensor(self.dataset["target"])
        else:
            self.graphs, self.labels = load_graphs(mat_path)

    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        save_info(info_path, {'num_classes': self.num_classes})
    
    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        self.num_classes = load_info(info_path)['num_classes']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)
    
    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 2

# GIN ARCHITECTURE

"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer


# Check for GPU and seed devices
torch.manual_seed(0)
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(0)
    print("*** WARNING *** Trying to use GPU: "+device)

# Load training data
train_dataset = LambdasDataset("ldata_train_6_23") # Make sure this is copied into ~/.dgl folder
train_dataset.load()
num_labels = train_dataset.num_labels
batch_size=1024
num_workers=1

# Create training dataloader
train_loader = GraphDataLoader(
    train_dataset,
    batch_size=batch_size,
    drop_last=False,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers)

# Load testing data
test_dataset = LambdasDataset("ldata_test_6_23") # Make sure this is copied into ~/.dgl folder
test_dataset.load()

# Create testing dataloader
test_loader = GraphDataLoader(
    test_dataset,
    batch_size=batch_size,
    drop_last=False,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers)

val_loader = test_loader


# Sanity check
print(train_dataset.graphs[0].ndata["data"])

# Setup model and training parameters
in_dim, mlp, feat_d, hid_d, n_classes, final_dropout, learn_eps, ag_node, ag_graph = 5, 2, 7, 64, 2, 0.5, False, "max", "sum"
model        = GIN(5, 2, 7, 32, 2, 0.5, False, "sum", "max").to(device) #params: in_dim, mlp, feature_d, hidden_d, n_classes, learn_eps, final_dropout, node_aggregation, graph aggregation (I think...)
lr           = 0.01
criterion    = nn.CrossEntropyLoss()  # default reduce is true
optimizer    = optim.Adam(model.parameters(), lr=lr)
scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
max_epochs   = 1
log_interval = 10
log_dir      = "./tb_logs/tmp/"
logs         = {'train':[], 'val':[]} #NOTE: For matplotlib plots
save_path    = "./torch_models"

# Make sure log/save directories exist
try:
    os.mkdir(log_dir)
except Exception:
    print("Could not create directory:",log_dir)
# !rm -rf ./torch_models

# Create trainer
def train_step(engine, batch):
    model.train()
    x, y   = batch
    x      = x.to(device)
    y      = y.to(device)
    y_pred = model(x,x.ndata["data"].float())
    loss   = criterion(y_pred, y)
    acc    = (y_pred.argmax(1) == y).type(torch.float).sum().item() / len(y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {'loss': loss.item(),
            'accuracy': acc,
            'y_pred': y_pred,
            'y': y}

trainer   = Engine(train_step)

# Add metrics
accuracy  = Accuracy(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
accuracy.attach(trainer, 'accuracy')
loss      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
loss.attach(trainer, 'loss')
roc_auc   = ROC_AUC(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
roc_auc.attach(trainer,'roc_auc')
roc_curve = RocCurve(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
roc_curve.attach(trainer,'roc_curve')

# Create validator
def val_step(engine, batch):
    model.eval()
    x, y   = batch
    x      = x.to(device)
    y      = y.to(device)
    y_pred = model(x,x.ndata["data"].float())
    loss   = criterion(y_pred, y)
    acc    = (y_pred.argmax(1) == y).type(torch.float).sum().item() / len(y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.train()
    return {'loss': loss.item(),
            'accuracy': acc,
#             'roc_curve' : roc_curve,
#             'roc_auc' : roc_auc
            'y_pred': y_pred,
            'y': y}

evaluator  = Engine(val_step)

# Add metrics
accuracy_  = Accuracy(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
accuracy_.attach(evaluator, 'accuracy')
loss_      = Loss(criterion,output_transform=lambda x: [x['y_pred'], x['y']])
loss_.attach(evaluator, 'loss')
roc_auc_   = ROC_AUC(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
roc_auc_.attach(evaluator,'roc_auc')
roc_curve_ = RocCurve(output_transform=lambda x: [x['y_pred'].argmax(1), x['y']])
roc_curve_.attach(evaluator,'roc_curve')

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(trainer):
    print(f"\rEpoch[{trainer.state.epoch} : " +
          f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
          f"Loss: {trainer.state.output['loss']:.3f} Accuracy: {trainer.state.output['accuracy']:.3f}",end='')

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    metrics = evaluator.run(train_loader).metrics
    logs['train'].append({metric:metrics[metric] for metric in metrics.keys()})
    print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.4f} Avg loss: {metrics['loss']:.4f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    metrics = evaluator.run(val_loader).metrics
    logs['val'].append({metric:metrics[metric] for metric in metrics.keys()})
    print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.4f} Avg loss: {metrics['loss']:.4f}")

# Create a TensorBoard logger
tb_logger = TensorboardLogger(log_dir=log_dir)

# Attach the logger to the trainer to log model's weights as a histogram after each epoch
tb_logger.attach(trainer,event_name=Events.EPOCH_COMPLETED,log_handler=WeightsHistHandler(model))

# Attach the logger to the trainer to log training loss at each iteration
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED,
    tag="training",
    output_transform=lambda loss: {"loss": loss["loss"]}
)
    
# Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
# We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
# of the `trainer` instead of `train_evaluator`.
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.EPOCH_COMPLETED,
    tag="training",
    metric_names=["loss","accuracy","roc_auc","roc_curve"],
    global_step_transform=global_step_from_engine(trainer),
)

# Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
# each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
# `trainer` instead of `evaluator`.
tb_logger.attach_output_handler(
    evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="validation",
    metric_names=["loss","accuracy","roc_auc","roc_curve"],
    global_step_transform=global_step_from_engine(evaluator)
)

# Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
tb_logger.attach_opt_params_handler(
    trainer,
    event_name=Events.ITERATION_STARTED,
    optimizer=optimizer,
    param_name='lr'  # optional
)

# Attach the logger to the trainer to log model's weights norm after each iteration
tb_logger.attach(
    trainer,
    event_name=Events.ITERATION_COMPLETED,
    log_handler=WeightsScalarHandler(model)
)

# Attach the logger to the trainer to log model's weights as a histogram after each epoch
tb_logger.attach(
    trainer,
    event_name=Events.EPOCH_COMPLETED,
    log_handler=WeightsHistHandler(model)
)

# Attach the logger to the trainer to log model's gradients norm after each iteration
tb_logger.attach(
    trainer,
    event_name=Events.ITERATION_COMPLETED,
    log_handler=GradsScalarHandler(model)
)

# Attach the logger to the trainer to log model's gradients as a histogram after each epoch
tb_logger.attach(
    trainer,
    event_name=Events.EPOCH_COMPLETED,
    log_handler=GradsHistHandler(model)
)

# Run training loop
trainer.run(train_loader, max_epochs=max_epochs)
tb_logger.close() #IMPORTANT!
torch.save(model.state_dict(), save_path)

# get ROC curve
pfn_fp, pfn_tp, threshs = np.array([el['roc_curve'][0] for el in logs['train']]), \
                          np.array([el['roc_curve'][1] for el in logs['train']]), \
                          np.array([el['roc_curve'][2] for el in logs['train']]),

# get area under the ROC curve
auc = np.array([el['roc_auc'] for el in logs['train']]),
print()
print('DGL AUC:', auc)
print()
f = plt.figure()

# some nicer plot settings 
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True

# plot the ROC curves
plt.plot(pfn_tp, 1-pfn_fp, '-', color='black', label='DGL')
# plt.plot(mass_tp, 1-mass_fp, '-', color='blue', label='Jet Mass')
# plt.plot(mult_tp, 1-mult_fp, '-', color='red', label='Multiplicity')

# axes labels
plt.xlabel('Lambda Event Efficiency')
plt.ylabel('Background Rejection')

# axes limits
plt.xlim(0, 1)
plt.ylim(0, 1)

# make legend and show plot
plt.legend(loc='lower left', frameon=False)
plt.show()
f.savefig("DGL_AUC.pdf")

##########################################################
# Plot loss and accuracy as a fn of epoch
f = plt.figure()
plt.clf()

# some nicer plot settings 
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True

# Get data and plot
loss_per_epoch = [el['loss'] for el in logs['train']]
acc_per_epoch  = [el['accuracy'] for el in logs['train']]
plt.plot([i for i in range(1,len(acc_per_epoch)+1)], loss_per_epoch, '-', color='tab:orange', label='loss')
plt.plot([i for i in range(1,len(acc_per_epoch)+1)], acc_per_epoch, '-', color='tab:blue', label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend(loc='lower left', frameon=False)
plt.show()
f.savefig("GIN_training_metrics.pdf")

##########################################################
# # Plot decisions
# bins = 100
# low = min(torch.min(p) for p in probs_Y[:,1])
# high = max(torch.max(p) for p in probs_Y[:,0])
# low_high = (low,high)
# f = plt.figure()
# plt.clf()
# # Plot training decisions
# print(probs_Y[:,0].detach().numpy())#DEBUGGING
# x1 = probs_Y[:,0].detach().numpy()
# x2 = np.ndarray(probs_Y[:,1].detach().numpy(),dtype=np.float32)
# plt.hist(x1, color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist1')
# plt.hist(x2.detach().numpy(), color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=True, label='hist2')
# plt.xlabel('output')
# plt.ylabel('counts')
# plt.show()
# f.savefig("DGL_decisions.pdf")