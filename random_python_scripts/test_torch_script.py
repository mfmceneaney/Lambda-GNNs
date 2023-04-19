import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys

sys.path.append('.')
from models_1_23_23 import GIN, MLP, Concatenate, MyModule

# Set Model Params
num_layers            = 3
num_mlp_layers        = 3
input_dim             = 7
hidden_dim            = 64
output_dim            = 2
final_dropout         = 0.5
learn_eps             = False
graph_pooling_type    = 'mean'
neighbor_pooling_type = 'mean'

my_module = GIN(
    num_layers,
    num_mlp_layers,
    input_dim,
    hidden_dim,
    output_dim,
    final_dropout,
    learn_eps,
    graph_pooling_type,
    neighbor_pooling_type
    )

# my_module = MyModule()

sm = torch.jit.script(my_module)
sm.save("scripted_module.pt")
