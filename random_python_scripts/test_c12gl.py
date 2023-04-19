# File I/O Imports
import hipopy.hipopy as hp
import uproot as ur
# import uproot3 as ur3

# Data Imports
import numpy as np
from numpy import ma
import awkward as ak
import pandas as pd

# !git clone https://github.com/mfmceneaney/c12gl.git
# import os
# os.chdir('c12gl')

import sys
sys.path.append('.')
print(sys.path)

#-------------------- DATA GENERATION --------------------#
def generate_data(
                filename      = "data.hipo",
                bank          = "NEW::bank",
                dtype         = "D",
                names         = ["px","py","pz"],
                namesAndTypes = {"px":"D","py":"D","pz":"D"},
                rows          = 7,
                nbatches      = 100,
                step          = 100,
                multiplier    = 1.0,
                offset        = 0.0
                ):
    # Open file
    file = hp.create(filename)
    file.newTree(bank,namesAndTypes)
    file.open() # IMPORTANT:  Open AFTER calling newTree, otherwise the banks will not be written!

    # Write batches of events to file
    for _ in range(nbatches):
        data = multiplier * np.random.random(size=(step,len(names),rows))
        data += offset
        file.extend({
            bank : data
        })

    file.close() # Can also use file.write()
    
# Generate data in `[0, 1)`
filename      = "data1.hipo"
bank          = "NEW::bank"
dtype         = "D"
names         = ["px","py","pz"]
namesAndTypes = {e:dtype for e in names}
rows          = 7 
nbatches = 100 # Choose a #
step = 100 # Choose a # (events per batch)
multiplier = 1.0
offset = 0.0

generate_data(
    filename=filename,
    bank=bank,
    dtype=dtype,
    names=names,
    namesAndTypes=namesAndTypes,
    rows=rows,
    nbatches=nbatches,
    step=step,
    multiplier=multiplier,
    offset=offset
)

# Generate data from another distribution
filename = "data2.hipo"
bank     = "NEW::bank"
dtype    = "D" #NOTE: For now all the bank entries have to have the same type.
names    = ["px","py","pz"]
namesAndTypes = {e:dtype for e in names}
rows = 7 # Chooose a #
nbatches = 100 # Choose a #
step = 100 # Choose a # (events per batch)
multiplier = 0.5
offset = 1.0

generate_data(
    filename=filename,
    bank=bank,
    dtype=dtype,
    names=names,
    namesAndTypes=namesAndTypes,
    rows=rows,
    nbatches=nbatches,
    step=step,
    multiplier=multiplier,
    offset=offset
)


#-------------------- DATASET CREATION --------------------#
from c12gl.preprocessing import *
from c12gl.dataloading import *
from tqdm import tqdm

# Define graph construction
def construct(nNodes,data):
    g = getWebGraph(nNodes)
    g.ndata['data'] = data
    return g

# Create DGL Dataset
name        = "test_dataset_9_30_22"
num_classes = 2
# ds          = GraphDataset(name=name,num_classes=num_classes)

# # Set parameters for first data distribution
# filename = "data1.hipo"
# bank     = "NEW::bank"
# step     = 100
# items    = ["px", "py", "pz"]
# keys     = [bank+"_"+item for item in items]

# # Create preprocessor and constructor
# p = Preprocessor(file_type="hipo")
# c = Constructor(construct=construct)

# # Define labels
# def getLabel(batch):
#     return [1 for arr in batch[bank+"_px"]]

# # Add labels to assign
# label_key = "ML::Label"
# p.addLabels({label_key:getLabel})

# # # Add processes with kwargs
# # for item in items:
# #     kwargs = {}
# #     p.addProcesses({bank+"_"+item:[normalize,kwargs]})

# # Loop files and build dataset
# for batch in tqdm(p.iterate(filename,banks=[bank],step=step)):
#     ls = batch[label_key]
#     datatensor = c.getDataTensor(batch,keys) #NOTE: Can filter datatensor for NaN here.
#     gs = c.getGraphs(datatensor) #NOTE: Check what is taking the longest here...
#     ds.extend(ls,gs)

# print("len(ds) = ",len(ds))

# # Set parameters for 2nd data distribution
# filename = "data2.hipo"
# bank     = "NEW::bank"
# step     = 10
# items    = ["px", "py", "pz"]
# keys     = [bank+"_"+item for item in items]

# # Create preprocessor and constructor
# p = Preprocessor(file_type="hipo")
# c = Constructor(construct=construct)

# # Define labels
# def getLabel(batch):
#     return [0 for arr in batch[bank+"_px"]]

# # Add labels to assign
# label_key = "ML::Label"
# p.addLabels({label_key:getLabel})

# # Add processes with kwargs
# for item in items:
#     p.addProcesses({bank+"_"+item:[normalize,{}]})

# # Loop files and build dataset
# for batch in tqdm(p.iterate(filename,banks=[bank],step=step)):
#     ls = batch[label_key]
#     datatensor = c.getDataTensor(batch,keys)
#     gs = c.getGraphs(datatensor)
#     ds.extend(ls,gs)

# print(len(ds))

#-------------------- MODEL CREATION --------------------#
# PyTorch Imports
import torch.nn as nn
import torch.nn.functional as F
import dgl

class SigmoidMLP(nn.Module):
    """
    SigmoidMLP with linear output
    """
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """SigmoidMLP layers construction

        Parameters
        ----------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(SigmoidMLP, self).__init__()
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
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.sigmoid(self.linear(x))
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.sigmoid(self.linears[-1](h))

from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, key = 'data'):
        g = dgl.add_self_loop(g)#NOTE: ADDED!
        h = g.ndata[key].float()
        h = self.conv1(g, h)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')
    
# DGL Graph Learning Imports
import dgl
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class MLP_SIGMOID(nn.Module):
    """MLP_SIGMOID with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP_SIGMOID layers construction

        Parameters
        ----------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP_SIGMOID, self).__init__()
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.sigmoid(self.linear(x))
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.sigmoid(self.linears[-1](h))


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Parameters
        ----------
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

        Parameters
        ----------
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

    def forward(self, g, key='data'):
        # list of hidden representation at each layer (including input)
        h = g.ndata[key].float()
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

    @property
    def name(self):
        """Name of model."""
        return "GIN"

class HeteroGIN(nn.Module):
    """GINHelp model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, input_dim2, hidden_dim2, n_final_mlp):
        """model parameters setting

        Parameters
        ----------
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
        super(HeteroGIN, self).__init__()
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
                    nn.Linear(input_dim, hidden_dim2-input_dim2))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, hidden_dim2-input_dim2))

        self.final_mlp = mlp = MLP(n_final_mlp, hidden_dim2, hidden_dim2, output_dim)

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, key='data', key2='kinematics'):
        # list of hidden representation at each layer (including input)
        h = g.ndata[key].float()
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
            score_over_layer += self.linears_prediction[i](pooled_h) # ORIGINALLY self.drop(self.linears_...)
            
        # Now you have data with shape [batch_size(can be smaller than requested on last batch!),num_classes]
        
        # Concatenate node data arrays along dim=-1 and take max to get kinematics
        h = []
        for u in dgl.unbatch(g):
            entry = u.ndata.get(key2).float()
            entry = torch.transpose(entry,0,1)
            entry = torch.max(entry,dim=-1).values
            entry = torch.reshape(entry,(1,entry.shape[0])) #NOTE: Relies on the array being 1D at this point.
            h.append(entry)
        h = torch.cat((h))
        h = torch.cat((score_over_layer,h),dim=-1)
        
        return self.drop(self.final_mlp(h))

    @property
    def name(self):
        """Name of model."""
        return "HeteroGIN"

#-------------------- TRAINING --------------------#
from ignite.engine import Engine, Events, EventEnum, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import global_step_from_engine, EarlyStopping
from ignite.contrib.handlers.mlflow_logger import MLflowLogger
import ignite.distributed as idist

import os

import matplotlib.pyplot as plt

def setPltParams(
    fontsize=20,
    axestitlesize=25,
    axeslabelsize=25,
    xticklabelsize=25,
    yticklabelsize=25,
    legendfontsize=25
    ):
    """
    Arguments
    ---------
    fontsize : int, default 20
    axestitlesize : int, default 25
    axeslabelsize : int, default 25
    xticklabelsize : int, default 25
    yticklabelsize : int, default 25
    legendfontsize : int, default 25

    Description
    -----------
    Set font sizes for matplotlib.plt plots.
    """
    plt.rc('font', size=20) #controls default text size                                                                                                                     
    plt.rc('axes', titlesize=25) #fontsize of the title                                                                                                                     
    plt.rc('axes', labelsize=25) #fontsize of the x and y labels                                                                                                            
    plt.rc('xtick', labelsize=20) #fontsize of the x tick labels                                                                                                            
    plt.rc('ytick', labelsize=20) #fontsize of the y tick labels                                                                                                            
    plt.rc('legend', fontsize=15) #fontsize of the legend

def train(
    rank,
    config,
    **kwargs
    ):

    """
    Parameters
    ----------
    rank : int
    config : dict

    Necessary entries in config
    ---------------------------
    model : torch.nn.Module, required
    device : str, required
    train_loader : dgl.dataloading.GraphDataloader, required
    val_loader : dgl.dataloading.GraphDataloader, required
    optimizer : torch.optim.optimizer, required
    scheduler : torch.optim.lr_scheduler, required
    criterion : torch.nn.loss, required
    max_epochs : int, required
    dataset : str, optional
        Default : ""
    prefix : str, optional
        Default : ""
    log_interval : int, optional
        Default : 10
    log_dir : str, optional
        Default : "logs/"
    save_path : str, optional
        Default : "model"
    verbose : bool, optional
        Default : True

    args,
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    max_epochs,
    dataset="",
    prefix="",
    log_interval=10,
    log_dir="logs/",
    save_path="model",
    verbose=True

    Returns
    -------
    logs : dict
        Dictionary of training and validation metric lists organized by epoch

    Description
    -----------
    Train a GNN using a basic supervised learning approach.
    """

    model = config['model']
    device = config['device']
    train_loader = config['train_loader']
    val_loader = config['val_loader']
    optimizer = config['optimizer']
    scheduler = config['scheduler']
    criterion = config['criterion']
    max_epochs = config['max_epochs']
#     dataset = config['dataset']
#     prefix = config['prefix']
    log_interval = config['log_interval']
    log_dir = config['log_dir']
    model_name = config['model_name']
    verbose = config['verbose']
    
    # Show model if requested
    if verbose: print(model)

    # Logs for matplotlib plots
    logs={'train':{'loss':[],'accuracy':[]}, 'val':{'loss':[],'accuracy':[]}}

    # Distributed setup
    dist = True
    if dist:
    
        import ignite.distributed as idist
        
        # Specific ignite.distributed
        if verbose: print(
            idist.get_rank(),
            ": run with config:",
            config,
            "- backend=",
            idist.backend(),
            "- world size",
            idist.get_world_size(),
        )

        device = idist.device()

        # Specific ignite.distributed
        train_loader = idist.auto_dataloader(
            train_loader.dataset,
            collate_fn=train_loader.collate_fn,
            batch_size=train_loader.batch_size,
            num_workers=train_loader.num_workers,
            shuffle=True,
            pin_memory=train_loader.pin_memory, #"cuda" in idist.device().type
            drop_last=train_loader.drop_last
        )
        val_loader = idist.auto_dataloader(
            val_loader.dataset,
            collate_fn=val_loader.collate_fn,
            batch_size=val_loader.batch_size,
            num_workers=val_loader.num_workers,
            shuffle=True,
            pin_memory=val_loader.pin_memory, #"cuda" in idist.device().type
            drop_last=val_loader.drop_last
        )

        # Model, criterion, optimizer setup
        print("DEBUGGING: model.parameters() = ",model.parameters())#DEBUGGING 
        model = idist.auto_model(model)
        print("DEBUGGING: idist.auto_model(model).parameters() = ",model.parameters())#DEBUGGING 
        optimizer = idist.auto_optim(optimizer)

    # Create train function
    def train_step(engine, batch):

        # Ensure model is in training mode
        model.train()

        # Get predictions and loss from data and labels
        x    = batch[0].to(device) # Batch data
        y    = batch[1][:,0].clone().detach().long().to(device) if len(np.shape(batch[1]))==2 else batch[1].clone().detach().long().to(device) #NOTE: This assumes labels is 2D and classification labels are integers
        prediction_raw = model(x) # Model prediction
        loss = criterion(prediction_raw, y) #NOTE: DO NOT APPLY SOFTMAX BEFORE CrossEntropyLoss

        # Step optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Apply softmax and get accuracy
        prediction_softmax = torch.softmax(prediction_raw, 1)
        prediction         = torch.max(prediction_softmax, 1)[1].view(-1, 1)
        acc                = (y.float().view(-1,1) == prediction.float()).sum().item() / len(y)

        return {
                'y': y,
                'prediction_raw': prediction_raw,
                'prediction': prediction,
                'loss': loss.detach().item(),
                'accuracy': acc
                }

    # Create validation function
    def val_step(engine, batch):

        # Ensure model is in evaluation mode
        model.eval()

        with torch.no_grad(): #NOTE: Important to call both model.eval and with torch.no_grad()! See https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval.

            # Get predictions and loss from data and labels
            x    = batch[0].to(device)
            y    = batch[1][:,0].clone().detach().long().to(device) if len(np.shape(batch[1]))==2 else batch[1].clone().detach().long().to(device) #NOTE: This assumes labels is 2D and classification labels are integers
            h    = model(x)
            prediction_raw = model(x) # Model prediction
            loss = criterion(prediction_raw, y)

        # Apply softmax and get accuracy
        prediction_softmax = torch.softmax(prediction_raw, 1)
        prediction         = torch.max(prediction_softmax, 1)[1].view(-1, 1)
        acc                = (y.float().view(-1,1) == prediction.float()).sum().item() / len(y)

        return {
                'y': y,
                'prediction_raw': prediction_raw,
                'prediction': prediction,
                'loss': loss.detach().item(),
                'accuracy': acc
                }

    # Create trainer
    trainer = Engine(train_step)

    # Add training metrics
    train_accuracy = Accuracy(output_transform=lambda x: [x['prediction'], x['y']])
    train_accuracy.attach(trainer, 'accuracy')
    train_loss     = Loss(criterion,output_transform=lambda x: [x['prediction_raw'], x['y']])
    train_loss.attach(trainer, 'loss')

    # Create evaluator
    evaluator = Engine(val_step)

    # Add evaluation metrics
    val_accuracy = Accuracy(output_transform=lambda x: [x['prediction'], x['y']])
    val_accuracy.attach(evaluator, 'accuracy')
    val_loss     = Loss(criterion,output_transform=lambda x: [x['prediction_raw'], x['y']])
    val_loss.attach(evaluator, 'loss')

#     # Set up early stopping
#     def score_function(engine):
#         val_loss = engine.state.metrics['loss']
#         return -val_loss

#     handler = EarlyStopping(
#         patience=patience,
#         min_delta=args.min_delta,
#         cumulative_delta=args.cumulative_delta,
#         score_function=score_function,
#         trainer=trainer
#         )
#     evaluator.add_event_handler(Events.COMPLETED, handler) #NOTE: The handler is attached to an evaluator which runs one epoch on validation dataset.

    
#     # Step learning rate #NOTE: DEBUGGING: TODO: Replace above...
#     @trainer.on(Events.EPOCH_COMPLETED)
#     def stepLR(trainer):
#         if type(scheduler)==torch.optim.lr_scheduler.ReduceLROnPlateau:
#             scheduler.step(trainer.state.output['loss'])#TODO: NOTE: DEBUGGING.... Fix this...
#         else:
#             scheduler.step()
            
    # Print training loss and accuracy
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def print_training_loss(trainer):
        if verbose: print(f"\rEpoch[{trainer.state.epoch}/{max_epochs} : " +
            f"{(trainer.state.iteration-(trainer.state.epoch-1)*trainer.state.epoch_length)/trainer.state.epoch_length*100:.1f}%] " +
            f"Loss: {trainer.state.output['loss']:.3f} Accuracy: {trainer.state.output['accuracy']:.3f}",end='')

    # Log training metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(trainer):
        metrics = evaluator.run(train_loader).metrics
        for metric in metrics.keys(): logs['train'][metric].append(metrics[metric])
        if verbose: print(f"\nTraining Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['loss']:.4f} Avg accuracy: {metrics['accuracy']:.4f}")

    # Log validation metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(trainer):
        metrics = evaluator.run(val_loader).metrics
        for metric in metrics.keys(): logs['val'][metric].append(metrics[metric])
        if verbose: print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['loss']:.4f} Avg accuracy: {metrics['accuracy']:.4f}")

    # Run training loop
    trainer.run(train_loader, max_epochs=max_epochs)
    if save_path!="":
        os.makedirs(log_dir, exist_ok=True)
        torch.save(model.to('cpu').state_dict(), os.path.join(log_dir,model_name+"_weights")) #NOTE: Save to cpu state so you can test more easily.
   

    #TODO: ADD MLFLOW LOGGER? ADD AS CONFIG ARGUMENT... THEN NEED TO HAVE SAME SETUP WHICH I THINK IS TRUE...
    # Create training/validation loss plot
    f = plt.figure()
    plt.subplot()
    plt.title('Loss per epoch')
    plt.plot(logs['train']['loss'],label="training")
    plt.plot(logs['val']['loss'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    f.savefig(os.path.join(log_dir,'training_loss.png'))

    # Create training/validation accuracy plot
    f = plt.figure()
    plt.subplot()
    plt.title('Accuracy per epoch')
    plt.plot(logs['train']['accuracy'],label="training")
    plt.plot(logs['val']['accuracy'],label="validation")
    plt.legend(loc='best', frameon=False)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    f.savefig(os.path.join(log_dir,'training_accuracy.png'))

    return logs

#-------------------- TRAINING --------------------#
from c12gl.dataloading import loadGraphDataset

dataset = "test_dataset_9_30_22"
split = 0.0
max_events = 2000
indices = [0,1600,1800,2000]
batch_size = 32
num_workers = 0
loaders = loadGraphDataset(
                            dataset=dataset,
                            prefix="",
                            key="data",
                            ekey="",
                            split=split,
                            max_events=max_events,
                            indices=indices,
                            batch_size=batch_size,
                            drop_last=False,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            verbose=True
                            )
train_loader, val_loader, eval_loader, nclasses, ndata_dim, edata_dim = loaders
print(loaders)

#-------------------- TRAINING --------------------#
import torch
import torch.optim as optim
device = torch.device('cpu')
num_layers = 5
num_mlp_layers = 3
input_dim = ndata_dim
hidden_dim = 32
output_dim = nclasses
final_dropout = 0.5
learn_eps = False
graph_pooling_type = 'mean'
neighbor_pooling_type = 'mean'
model = GIN(num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type).to(device)
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
gamma = 0.1
patience = 10
threshold = 0.001
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                                optimizer,
                                                mode='min',
                                                factor=gamma,
                                                patience=patience,
                                                threshold=threshold,
                                                threshold_mode='rel',
                                                cooldown=0,
                                                min_lr=0,
                                                eps=1e-08,
                                                verbose=True
                                                )
criterion = nn.CrossEntropyLoss()
max_epochs = 25
dataset = "test_dataset_9_30_22"
prefix = ""
log_interval = 10
log_dir = "test_log_dir"
model_name = 'model'
verbose = True

rank = 0
config = {
    'model'        : model,
    'device'       : torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0'),
    'train_loader' : train_loader,
    'val_loader'   : val_loader,
    'optimizer'    : optimizer,
    'scheduler'    : scheduler,
    'criterion'    : criterion,
    'max_epochs'   : max_epochs,
    'dataset'      : dataset,
    'prefix'       : prefix,
    'log_interval' : log_interval,
    'log_dir'      : log_dir,
    'model_name'   : model_name,
    'verbose'      : verbose
}

train(rank,config)


# backend = "nccl"  # torch native distributed configuration on multiple GPUs
# # backend = "xla-tpu"  # XLA TPUs distributed configuration
# # backend = None  # no distributed configuration
# #
# dist_configs = {'nproc_per_node': 4}  # Use specified distributed configuration if launch as python main.py
# # dist_configs["start_method"] = "fork"  # Add start_method as "fork" if using Jupyter Notebook
# with idist.Parallel(backend=backend, **dist_configs) as parallel:
#     parallel.run(train, config, myrandomvariable=None)