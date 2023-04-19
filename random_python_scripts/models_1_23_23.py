###############################
# Matthew McEneaney
# 7/27/21
###############################

from __future__ import absolute_import, division, print_function

# DGL Graph Learning Imports
import dgl
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------- Models -------------------------#

# Test model
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, (l1, l2) in enumerate(zip(self.linears,self.linears)):
            if i < len(self.linears) - 1: x = l1(x) + l2(x)
            else: x = l1(x)
        return x

# Concatenation model
class Concatenate(nn.Module):
    def __init__(self,models,name="Concatenate"):
        super(Concatenate, self).__init__()
        self.models = nn.ModuleList(models)
        self.name = name
    
    def forward(self,h):
        for m in self.models: #NOTE: Make sure if you decide to copy arrays you put them on the correct device!
            h = m(h)
        return h

    # @property
    def name(self):
        """
        Name of model.
        """
        return self.name

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
        # h = self.bn(h)
        # h = F.relu(h)
        return h

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
        self.linear = nn.Linear(input_dim, output_dim)
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)])
        # self.final_linear = nn.Linear(hidden_dim, output_dim)

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
            # self.final_linear = nn.Linear(hidden_dim, output_dim) #NOTE: CHANGED FROM APPEND TO COMPLETELY SEPARATE LAYER

            # for layer in range(num_layers - 1): #NOTE CHANGED TO BELOW DEBUGGING
            for layer in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x #NOTE: OLD
            for i, (bn, ln) in enumerate(zip(self.batch_norms,self.linears)):
                if i < self.num_layers - 1: h = F.relu(bn(ln(h)))
                # else: return ln(h)
                else: return F.relu(bn(ln(h)))
            # return self.final_linear(h)

            # # If MLP
            # h = x #NOTE: OLD
            # for i, (bn, ln) in enumerate(zip(self.batch_norms,self.linears)):
            #     if i < self.num_layers - 1: h = F.relu(self.batch_norms[str(i)](self.linears[str(i)](h)))
            # return self.linears[-1](h)

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
